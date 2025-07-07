use crate::{Error, Topology};
use crate::imbalance::imbalance;
use std::collections::HashSet;
use std::collections::HashMap;
use num_traits::ToPrimitive;
use rand::{thread_rng, Rng};
use rayon::prelude::*;


#[derive(Debug)]
struct Move {
    // Struct to store data about a move that can either lead to better edge cuts or
    // re-balance the weights.

    //The index of the vertex.
    vertex: usize,

    // The partition ID of the partition wherr the vertex should move to.
    partition_id: usize
}

fn jet_partitioner<T>(
    partition: &mut [usize],
    weights: &[f64],
    adjacency: T,
    iterations: u32,
    balance_factor: f64,
    filter_ratio: f64,
    tolerance_factor: f64,
)
where
    T: Topology<i64> + Sync, {

    debug_assert!(!partition.is_empty());
    debug_assert_eq!(partition.len(), weights.len());
    debug_assert_eq!(partition.len(), adjacency.len());

    let mut partition_best = partition.to_vec();
    let mut partition_iter = partition.to_vec();
    let mut current_iteration = 0;
    let num_of_partitions = partition.iter().collect::<HashSet<_>>().len();
    let mut vertex_connectivity_data_structure = init_vertex_connectivity_data_structure(&adjacency,
                                                                                     partition);
    let mut locked_vertices = HashSet::new();

    while current_iteration < iterations {
        let moves;
        if imbalance(num_of_partitions, &partition_iter, weights.par_iter().cloned()) < balance_factor {
            // the jetlp subroutine is used to generate better a better partition
            moves = jetlp(&adjacency,
                          &partition_iter,
                          &vertex_connectivity_data_structure,
                          &locked_vertices,
                          filter_ratio);

            // Based on the suggested moves of the jetlp subroutine, we lock the vertices to ensure
            // that they don't become eligible to move in the next iteration. This prevents oscillation
            // of vertices
            locked_vertices = get_locked_vertices(&moves);
        } else {
            // the jetrw subroutine is run to balance the weights of the partition
            // (should the partitions weights become highly imbalanced)
            moves = jetrw(&adjacency,
                          &partition_iter,
                          weights,
                          &vertex_connectivity_data_structure,
                          num_of_partitions,
                          balance_factor);
        }

        // The moves from either jetlp or jetrw are applied on the current partition state.
        update_parts_and_vertex_connectivity(&adjacency,
                                             &mut partition_iter,
                                             &mut vertex_connectivity_data_structure,
                                             moves);

        // Check if the current iteration partition is balancee
        if imbalance(num_of_partitions, &partition_iter, weights.par_iter().cloned()) < balance_factor {
            // Check if the current iteration partition is better than the current best partition
            if adjacency.edge_cut(&partition_iter) < adjacency.edge_cut(&partition_best) {
                // Current iteration partition is chosen as the best partition
                if adjacency.edge_cut(&partition_iter).to_f64().unwrap() < tolerance_factor*adjacency.edge_cut(&partition_iter).to_f64().unwrap(){
                    current_iteration = 0;
                }
                partition_best = partition_iter.to_vec();
            } else {
                current_iteration += 1;
            }
        } else if imbalance(num_of_partitions,
                            &partition_iter,
                            weights.par_iter().cloned())
            <
            imbalance(num_of_partitions,
                      &partition_best,
                      weights.par_iter().cloned()) {
            // Current iteration is better balanced than the best iteration, hence we make this
            // the best iteration
            partition_best = partition_iter.to_vec();
            current_iteration = 0
        } else {
            current_iteration += 1;
        }
    }

    partition.copy_from_slice(&partition_best);
}

fn jetlp<T>(graph: &T, partition: &[usize], vertex_connectivity_data_structure: &Vec<HashMap<usize, i64>>, locked_vertices: &HashSet<usize>, filter_ratio: f64) -> Vec<Move>
where
    T: Topology<i64> + Sync{
    let mut partition_dest = partition.to_vec();
    let mut gain = vec![0; graph.len()];

    // iterate over all the vertices to find out which vertices provides the best gain (decrease in edge cut)
    for vertex in 0..graph.len() {

        if !locked_vertices.contains(&vertex) {

            // Stores the neighbors of the vertex that belong to different partition as that of the vertex.
            let mut neighbors_eligible_partitions = Vec::new();

            for (neighbor_vertex, _) in graph.neighbors(vertex) {
                if partition[neighbor_vertex] != partition[vertex] {
                    neighbors_eligible_partitions.push(partition[neighbor_vertex]);
                }
            }

            // If the vertex has neighbors that belong to different vertices we calculate the gain
            // to find out which of them would cause the edge cut to become better.
            if neighbors_eligible_partitions.len() != 0 {
                partition_dest[vertex] = get_most_connected_partition(vertex,
                                                                      &neighbors_eligible_partitions,
                                                                      vertex_connectivity_data_structure);

                gain[vertex] = conn(vertex, partition_dest[vertex],
                                    &vertex_connectivity_data_structure)
                    - conn(vertex, partition[vertex],
                           &vertex_connectivity_data_structure);
            }
        }
    }

    // We apply a filter to check which of the vertices are eligible for moving from one partition
    // to another. Either the gain should be positive or can be slightly negative (based on the filter ratio).
    // Slightly negative gain vertices are also considered in the hope that they could provide better global solutions
    let first_filter_eligible_moves = gain_conn_ratio_filter(
                                                             locked_vertices,
                                                             partition,
                                                             &gain,
                                                             vertex_connectivity_data_structure,
                                                             filter_ratio);

    // We now try to approximate the true gain that would occur as two positive moves can when
    // applied simultaneously can be detrimental.
    let mut gain2:HashMap<usize, i64> = HashMap::new();

    for &vertex in &first_filter_eligible_moves {

        for (neighbor_vertex, edge_weight) in graph.neighbors(vertex){
            let mut partition_source = partition[neighbor_vertex];

            if is_higher_placed(neighbor_vertex, vertex, &gain, &first_filter_eligible_moves) {
                partition_source = partition_dest[neighbor_vertex];
            }

            if partition_source == partition_dest[vertex] {
                *gain2.entry(vertex).or_insert(0) += edge_weight;
            } else if partition_source == partition[vertex]{
                *gain2.entry(vertex).or_insert(0) -= edge_weight;
            }
        }
    }

    // From the newly calculated approximate gain values, we generate moves that yield positive gain.
    non_negative_gain_filter(&first_filter_eligible_moves, &partition_dest, &gain2)
}

// fn jetlp_rayon<T>(graph: &T, partition: &[usize], vertex_connectivity_data_structure: &Vec<HashMap<usize, i64>>, locked_vertices: &HashSet<usize>, num_partitions: usize, filter_ratio: f64) -> Vec<Move>
// where
//     T: Topology<i64> + Sync{
//     //let mut partition_dest = partition.to_vec();
//     //let mut gain = vec![0; graph.len()];
//
//     // iterate over all the vertices to find out which vertices provides the best gain (decrease in edge cut)
//     let (partition_dest, gain): (Vec<usize>, Vec<i64>) = (0..graph.len()).into_par_iter().map(|vertex| {
//         let mut partition_dest = partition[vertex];
//         let mut calculated_gain = 0.0;
//         if !locked_vertices.contains(&vertex) {
//
//             // Stores the neighbors of the vertex that belong to different partition as that of the vertex.
//             let mut neighbors_eligible_partitions = Vec::new();
//
//             for (neighbor_vertex, _) in graph.neighbors(vertex) {
//                 if partition[neighbor_vertex] != partition[vertex] {
//                     neighbors_eligible_partitions.push(partition[neighbor_vertex]);
//                 }
//             }
//
//             // If the vertex has neighbors that belong to different vertices we calculate the gain
//             // to find out which of them would cause the edge cut to become better.
//             if neighbors_eligible_partitions.len() != 0 {
//                 let partition_dest = get_most_connected_partition(vertex,
//                                                                       &neighbors_eligible_partitions,
//                                                                       vertex_connectivity_data_structure);
//
//                 let calculated_gain = conn(vertex, partition_dest,
//                                     &vertex_connectivity_data_structure)
//                     - conn(vertex, partition[vertex],
//                            &vertex_connectivity_data_structure);
//             }
//         }
//         (partition_dest, calculated_gain)
//     }).unzip();
//
//     // We apply a filter to check which of the vertices are eligible for moving from one partition
//     // to another. Either the gain should be positive or can be slightly negative (based on the filter ratio).
//     // Slightly negative gain vertices are also considered in the hope that they could provide better global solutions
//     let first_filter_eligible_moves = gain_conn_ratio_filter(
//                                                              locked_vertices,
//                                                              partition,
//                                                              &gain,
//                                                              vertex_connectivity_data_structure,
//                                                              filter_ratio);
//
//     // We now try to approximate the true gain that would occur as two positive moves can when
//     // applied simultaneously can be detrimental.
//     let mut gain2:HashMap<usize, i64> = HashMap::new();
//
//     first_filter_eligible_moves.par_iter().for_each(|&vertex| {
//
//         for (neighbor_vertex, edge_weight) in graph.neighbors(vertex){
//             let mut partition_source = partition[neighbor_vertex];
//
//             if is_higher_placed(neighbor_vertex, vertex, &gain, &first_filter_eligible_moves) {
//                 partition_source = partition_dest[neighbor_vertex];
//             }
//
//             if partition_source == partition_dest[vertex] {
//                 *gain2.entry(vertex).or_insert(0) += edge_weight;
//             } else if partition_source == partition[vertex]{
//                 *gain2.entry(vertex).or_insert(0) -= edge_weight;
//             }
//         }
//     });
//
//     // From the newly calculated approximate gain values, we generate moves that yield positive gain.
//     non_negative_gain_filter(&first_filter_eligible_moves, &partition_dest, &gain2)
// }

fn jetrw<T>(graph: &T, partitions: &[usize], vertex_weights: &[f64], vertex_connectivity_data_structure: &Vec<HashMap<usize, i64>>, num_partitions: usize, balance_factor: f64) -> Vec<Move>
where
    T: Topology<i64> + Sync {

    let max_slots: usize = 25;
    let mut partitions_dest = partitions.to_vec();
    let total_weight: f64 = vertex_weights.iter().cloned().sum();
    let max_weight_per_partitions = (1f64 + balance_factor)*total_weight/(num_partitions as f64);
    let mut loss = vec![0; partitions.len()];
    let num_of_vertices = graph.len();
    let mut heavy_partitions: Vec<usize> = Vec::new();
    let mut light_partitions: Vec<usize> = Vec::new();

    // We set what the max weight of the destination partition can be.
    // This is to prevent oscillations when the jetrw algorithm is rerun
    let mut max_weight_dest = max_weight_per_partitions*0.99;
    if max_weight_dest < max_weight_per_partitions - 100f64 {
        max_weight_dest = max_weight_per_partitions - 100f64;
    }

    // We find out what the partitions are heavy (need to be downsized) and what partitions are light
    // (can act as valid destination partitions).
    for partition in 0..num_partitions{

        if max_weight_per_partitions < get_weight_of_partition(partition, partitions, vertex_weights) {
            heavy_partitions.push(partition);
        }

        if max_weight_dest >= get_weight_of_partition(partition, partitions, vertex_weights) {
            light_partitions.push(partition);
        }
    }

    // We find out the loss for each eligible vertex move (from an overwight partition to an underweight partition).
    // A positive loss indicates an increase in edge cut.
    for vertex in 0..num_of_vertices{
        let weight_of_partition = get_weight_of_partition(partitions[vertex],
                                                          partitions,
                                                          vertex_weights);
        let limit = 1.5*(weight_of_partition - ((total_weight)/(num_partitions as f64)));

        if heavy_partitions.contains(&partitions[vertex]) && (vertex_weights[vertex]) < limit {
            let adjacent_partitions = &get_adjacent_eligible_destination_partitions(
                graph,
                vertex,
                &partitions,
                &light_partitions);

            if adjacent_partitions.len() == 0{
                partitions_dest[vertex] = light_partitions[thread_rng().gen_range(0..light_partitions.len())];
            } else {
                partitions_dest[vertex] = get_most_connected_partition(vertex,
                                                                       adjacent_partitions,
                                                                       vertex_connectivity_data_structure);
            }
            loss[vertex] = conn(vertex,
                                partitions[vertex],
                                vertex_connectivity_data_structure) -
                           conn(vertex,
                                partitions_dest[vertex],
                                vertex_connectivity_data_structure);
        }
    }

    // We slot the loss values into different buckets. This is to prevent sorting the loss values
    // which can be expensive.
    let mut bucket = init_bucket(heavy_partitions.len(), max_slots);

    for vertex in 0..num_of_vertices{

        if heavy_partitions.contains(&partitions[vertex]) {
            let index = heavy_partitions.iter().position(|&x| x == partitions[vertex]).unwrap();
            let slot = calculate_slot(loss[vertex], max_slots);
            bucket[get_index_for_bucket(index, slot, max_slots)].push(vertex);
        }
    }

    // For each of the heavy partitions we decide the vertices that can be moved from the
    // heavy partitions such that the increase in edge cut is minimized.
    let mut moves = Vec::new();
    for (index, &heavy_partition) in heavy_partitions.iter().enumerate(){
        let mut m = 0f64;
        let m_max = get_weight_of_partition(heavy_partition,
                                            partitions,
                                            vertex_weights) - max_weight_per_partitions;

        for slot in 0..max_slots {

            for &vertex in &bucket[get_index_for_bucket(index, slot, max_slots)] {
                m = m + (vertex_weights[vertex]);

                if m < m_max {
                    moves.push(Move{vertex, partition_id: partitions_dest[vertex]});
                }
            }
        }
    }

    moves
}

fn get_locked_vertices(moves: &Vec<Move>) -> HashSet<usize> {
    // This function gets the list of locked vertices that shouldn't be moved in the subsequent iterations.

    let mut locked_vertices = HashSet::new();

    for single_move in moves{
        locked_vertices.insert(single_move.vertex);
    }

    locked_vertices
}

fn gain_conn_ratio_filter(locked_vertices: &HashSet<usize>, partitions: &[usize], gain: &[i64], vertex_connectivity_data_structure: &Vec<HashMap<usize, i64>>, filter_ratio: f64) -> Vec<usize> {
    // Get a list of vertices that have a positive gain or slightly negative gain value (based on the filter ratio).

    let num_vertices = partitions.len();
    let mut list_of_moveable_vertices  = Vec::new();

    for vertex in 0..num_vertices {
        if !locked_vertices.contains(&vertex)
            &&
            (gain[vertex] > 0 || -gain[vertex] < (filter_ratio * (conn(vertex,
                                                                       partitions[vertex],
                                                                       vertex_connectivity_data_structure) as f64))
                                                                       .floor() as i64){
            list_of_moveable_vertices.push(vertex);
        }
    }

    list_of_moveable_vertices
}

fn non_negative_gain_filter(first_filter_eligible_moves: &[usize],
                            partition_dest: &[usize],
                            gain: &HashMap<usize, i64>) -> Vec<Move> {
    // Gets the list of moves that have positive gain.
    let mut list_of_moves: Vec<Move> = Vec::new();

    for vertex in first_filter_eligible_moves {
        if gain[vertex] > 0 {
            list_of_moves.push(Move{vertex: *vertex, partition_id: partition_dest[*vertex]});
        }
    }

    list_of_moves
}

fn conn(vertex_id: usize,
        partition_id: usize,
        vertex_connectivity_data_structure: &Vec<HashMap<usize, i64>>) -> i64 {
    // Gets how well a vertex is connected to a partition (adds all the edge weights connected to the partition).

    *vertex_connectivity_data_structure[vertex_id].get(&partition_id).unwrap_or(&0)
}

fn get_most_connected_partition(
    vertex_id: usize,
    partition_ids: &[usize],
    vertex_connectivity_data_structure: &Vec<HashMap<usize, i64>>) -> usize {
    // Get the most connected partition to a particular vertex.

    let mut connections = i64::MIN;
    let mut most_connected_partition = partition_ids[0];

    for partition_id in partition_ids {

        if vertex_connectivity_data_structure[vertex_id][partition_id] > connections {
            connections = vertex_connectivity_data_structure[vertex_id][partition_id];
            most_connected_partition = *partition_id;
        }
    }
    most_connected_partition
}

fn init_vertex_connectivity_data_structure<T>(graph: &T,
                                              partition: &[usize]) -> Vec<HashMap<usize, i64>>
where
    T: Topology<i64> + Sync {
    // Initialize the vertex connectivity data structure.

    let mut vertex_connectivity_data_structure = vec![HashMap::new(); partition.len()];

    let num_of_vertices = graph.len();

    for vertex in 0..num_of_vertices {

        let neighbours = graph.neighbors(vertex);
        for (neighbour_vertex, edge_weight) in neighbours {
            *vertex_connectivity_data_structure[vertex]
                .entry(partition[neighbour_vertex])
                .or_insert(0) += edge_weight;
        }
    }

    vertex_connectivity_data_structure
}

fn update_parts_and_vertex_connectivity<T>(
    graph: &T,
    partition: &mut [usize],
    vertex_connectivity_data_structure: &mut Vec<HashMap<usize, i64>>,
    moves: Vec<Move>)
where
    T: Topology<i64> + Sync {
    // Updates the partitions and the vertex connectivity data structure using the given list of moves.

    for single_move in &moves {
        let vertex = single_move.vertex;
        let partition_source = partition[vertex];

        for (neighbour_vertex, edge_weight) in graph.neighbors(vertex) {
            let vertex_connectivity_hashmap = &mut vertex_connectivity_data_structure[neighbour_vertex];
            *vertex_connectivity_hashmap
                .entry(partition_source)
                .or_insert(0) -= edge_weight;

            if vertex_connectivity_hashmap[&partition_source] == 0{
                vertex_connectivity_hashmap.remove(&partition_source);
            }
        }

        partition[vertex] = single_move.partition_id;
    }

    for single_move in &moves {
        let vertex = single_move.vertex;
        let partition_dest = single_move.partition_id;

        for (neighbour_vertex, edge_weight) in graph.neighbors(vertex) {
            let vertex_connectivity_hashmap = &mut vertex_connectivity_data_structure[neighbour_vertex];
            *vertex_connectivity_hashmap.entry(partition_dest).or_insert(0) += edge_weight;
        }
    }
}

fn is_higher_placed(vertex1: usize, vertex2: usize, gain: &[i64], list_of_vertices: &[usize]) -> bool {
    // Checks if vertex1 is better ranked than vertex2 (used in the vertex afterburner).

    if list_of_vertices.contains(&vertex1) && (gain[vertex1] > gain[vertex2] || (gain[vertex1] == gain[vertex2] && vertex1 < vertex2)){
        return true;
    }

    false
}

fn calculate_slot(loss: i64, max_slot_size: usize) -> usize {
    // Calculate the slot in which the vertex should be put in based on the loss value.

    if loss < 0 {
        0
    } else if loss == 0 {
        1
    } else {
        ((2 + loss.ilog2()) as usize).min(max_slot_size)
    }
}

fn get_adjacent_eligible_destination_partitions<T>(
    graph: &T,
    vertex: usize,
    partitions: &[usize],
    eligible_partitions: &[usize]) -> Vec<usize>
where
    T: Topology<i64> + Sync{
    // Gets the list of partitions belong to the neighbors of a particular vertex.

    let mut adjacent_eligible_partitions = Vec::new();

    for (neighbour, _) in graph.neighbors(vertex){

        if eligible_partitions.contains(&partitions[neighbour]) {
            adjacent_eligible_partitions.push(partitions[neighbour]);
        }
    }
    adjacent_eligible_partitions
}

fn get_weight_of_partition(partition_id: usize, partitions: &[usize], vertex_weights: &[f64]) -> f64 {
    // Gets the weight of a particular partition.

    let mut weight = 0f64;

    for (index, partition) in partitions.iter().enumerate() {

        if partition == &partition_id {
            weight += vertex_weights[index] as f64;
        }
    }

    weight
}

fn get_index_for_bucket(partition_index: usize, slot: usize, max_slots: usize) -> usize {
    // Gets the index of the bucket based on slot and partition index.

    partition_index * max_slots + slot
}

fn init_bucket(num_heavy_partitions: usize, max_slots: usize) -> Vec<Vec<usize>>{
    // Initialize the bucket where a list of vertices are stored in slots.

    let rows = num_heavy_partitions*max_slots;
    let mut bucket: Vec<Vec<usize>> = Vec::with_capacity(rows);

    for _ in 0..rows {
        bucket.push(Vec::new());
    }

    bucket
}

/// Jet Partitioner
///
/// An implementation of the Jet Partitioner algorithm
/// for graph partition refinement.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), coupe::Error> {
/// use coupe::Partition as _;
/// use coupe::Point2D;
///
/// //    swap
/// // 0  1  0  1
/// // +--+--+--+
/// // |  |  |  |
/// // +--+--+--+
/// // 0  0  1  1
/// let points = [
///     Point2D::new(0., 0.),
///     Point2D::new(1., 0.),
///     Point2D::new(2., 0.),
///     Point2D::new(3., 0.),
///     Point2D::new(0., 1.),
///     Point2D::new(1., 1.),
///     Point2D::new(2., 1.),
///     Point2D::new(3., 1.),
/// ];
/// let weights = [1.0; 8];
/// let mut partition = [0, 0, 1, 1, 0, 1, 0, 1];
///
/// let mut adjacency = sprs::CsMat::empty(sprs::CSR, 0);
/// adjacency.insert(0, 1, 1);
/// adjacency.insert(1, 2, 1);
/// adjacency.insert(2, 3, 1);
/// adjacency.insert(4, 5, 1);
/// adjacency.insert(5, 6, 1);
/// adjacency.insert(6, 7, 1);
/// adjacency.insert(0, 4, 1);
/// adjacency.insert(1, 5, 1);
/// adjacency.insert(2, 6, 1);
/// adjacency.insert(3, 7, 1);
///
/// // symmetry
/// adjacency.insert(1, 0, 1);
/// adjacency.insert(2, 1, 1);
/// adjacency.insert(3, 2, 1);
/// adjacency.insert(5, 4, 1);
/// adjacency.insert(6, 5, 1);
/// adjacency.insert(7, 6, 1);
/// adjacency.insert(4, 0, 1);
/// adjacency.insert(5, 1, 1);
/// adjacency.insert(6, 2, 1);
/// adjacency.insert(7, 3, 1);
///
///
///
/// coupe::JetPartitioner { ..Default::default() }
///     .partition(&mut partition, (adjacency.view(), &weights))?;
///
/// assert_eq!(partition, [0, 0, 1, 1, 0, 0, 1, 1]);
/// # Ok(())
/// # }
/// ```
///
/// # Reference
///
/// Gilbert, Michael S., et al. "Jet: Multilevel graph partitioning on graphics processing units."
/// SIAM Journal on Scientific Computing 46.5 (2024): B700-B724.

#[derive(Debug, Clone, Copy)]
pub struct JetPartitioner {
    /// This indicates the number of times jetlp/jetrw combination should run without seeing
    /// any improvement before terminating the algorithm
    pub iterations: u32,

    /// A numerical factor ranging between 0.0 and 1.0 that determines the maximum allowable
    /// deviation for a partition. The maximum weight of a partition with a balance factor of lambda
    /// can be (1+lambda)*((totol weight of graph)/(number of partitions)).
    pub balance_factor: f64,

    /// A numerical ratio ranging from 0.0 to 1.0 that determines which vertices are eligible for consideration based on
    /// their gain value in the first filter. A vertice would be considered
    /// if -gain(vertice) > (filter ratio)*(maximum connectivity of the vertice to any destination partition)
    pub filter_ratio: f64,

    /// A nymerical factor ranging from 0.0 to 1.0 that is used to determine when to reset the iteration counter.
    /// If the new edge cut is less than tolerance factor times the best edge cut, then the
    /// iteration counter would be reset, otherwise the iteration counter would increment
    /// as it indicates the edge is becoming better at a very slow pace.
    pub tolerance_factor: f64,
}

impl Default for JetPartitioner {
    fn default() -> Self {
        JetPartitioner {
            iterations: 12,
            balance_factor: 0.1,
            filter_ratio: 0.75,
            tolerance_factor: 0.99,
        }
    }
}
impl<'a, T> crate::Partition<(T, &'a [f64])> for JetPartitioner
where
    T: Topology<i64> + Sync,
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (adjacency, weights): (T, &'a [f64]),
    ) -> Result<Self::Metadata, Self::Error> {

        if part_ids.len() != weights.len() {
            return Err(Error::InputLenMismatch {
                expected: part_ids.len(),
                actual: weights.len(),
            });
        }
        if part_ids.len() != adjacency.len() {
            return Err(Error::InputLenMismatch {
                expected: part_ids.len(),
                actual: adjacency.len(),
            });
        }
        let metadata = jet_partitioner(
            part_ids,
            weights,
            adjacency,
            self.iterations,
            self.balance_factor,
            self.filter_ratio,
            self.tolerance_factor,
        );
        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_locked_vertices() {
        // Arrange
        let moves = vec![Move{vertex:0, partition_id:3},
                                     Move{vertex:3, partition_id:4},
                                     Move{vertex:4, partition_id:5}];

        // Act
        let locked_vertices = get_locked_vertices(&moves);

        // Assert
        assert!(locked_vertices.contains(&(0usize)));
        assert!(locked_vertices.contains(&(3usize)));
        assert!(locked_vertices.contains(&(4usize)));
        assert!(!locked_vertices.contains(&(2usize)));

    }
    #[test]
    fn test_init_vertex_connectivity_data_structure() {
        // Arrange
        let mut adjacency = sprs::CsMat::empty(sprs::CSR, 0);
        adjacency.insert(0, 1, 2);
        adjacency.insert(0, 2, 1);
        adjacency.insert(0, 3, 4);
        adjacency.insert(1, 0, 2);
        adjacency.insert(2, 0, 1);
        adjacency.insert(3, 0, 4);

        let partition = [0, 0, 0, 1];

        // Act
        let vtx_conn_data_struct = init_vertex_connectivity_data_structure(
            &adjacency.view(),
            &partition);

        // Assert
        assert_eq!(*vtx_conn_data_struct[0].get(&0).unwrap(), 3);
        assert_eq!(*vtx_conn_data_struct[0].get(&1).unwrap(), 4);

    }

    #[test]
    fn test_get_most_connected_partition(){
        // Arrange
        let mut adjacency = sprs::CsMat::empty(sprs::CSR, 0);
        adjacency.insert(0, 1, 2);
        adjacency.insert(0, 2, 1);
        adjacency.insert(0, 3, 4);
        adjacency.insert(1, 0, 2);
        adjacency.insert(2, 0, 1);
        adjacency.insert(3, 0, 4);

        let partition = [0, 0, 0, 1];
        let vtx_conn_data_struct = init_vertex_connectivity_data_structure(
            &adjacency.view(),
            &partition);

        // Act
        let most_connected_partition = get_most_connected_partition(
            0,
            &partition,
            &vtx_conn_data_struct);

        // Assert
        assert_eq!(most_connected_partition, 1);
    }

    #[test]
    fn test_conn() {
        // Arrange
        let mut adjacency = sprs::CsMat::empty(sprs::CSR, 0);
        adjacency.insert(0, 1, 2);
        adjacency.insert(0, 2, 1);
        adjacency.insert(0, 3, 4);
        adjacency.insert(1, 0, 2);
        adjacency.insert(2, 0, 1);
        adjacency.insert(3, 0, 4);

        let partition = [0, 0, 0, 1];
        let vtx_conn_data_struct = init_vertex_connectivity_data_structure(
            &adjacency.view(),
            &partition);

        // Act
        let conn_strength_part_0 = conn(0, 0, &vtx_conn_data_struct);
        let conn_strength_part_1 = conn(0, 1, &vtx_conn_data_struct);

        // Assert
        assert_eq!(conn_strength_part_0, 3);
        assert_eq!(conn_strength_part_1, 4);

    }

    #[test]
    fn test_non_negative_gain_filter() {
        // Arrange
        let mut gain = HashMap::new();
        gain.insert(0, 3);
        gain.insert(1, 2);
        gain.insert(2, -1);
        let eligible_vertices_to_move = [0, 2];
        let partition_dest  = [1, 0, 1];

        // Act
        let moves = non_negative_gain_filter(
            &eligible_vertices_to_move,
            &partition_dest,
            &gain);

        // Assert
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].vertex, 0);
        assert_eq!(moves[0].partition_id, 1);
    }

    #[test]
    fn test_gain_conn_ratio_filter() {
        // Arrange
        let mut adjacency = sprs::CsMat::empty(sprs::CSR, 0);
        adjacency.insert(0, 1, 3);
        adjacency.insert(0, 2, 1);
        adjacency.insert(0, 3, 4);
        adjacency.insert(1, 0, 3);
        adjacency.insert(2, 0, 1);
        adjacency.insert(3, 0, 4);

        let partitions = [0, 0, 0, 1];
        let vtx_conn_data_struct = init_vertex_connectivity_data_structure(
            &adjacency.view(),
            &partitions);
        let gain = [-1, 2, -2, -2];
        let filter_ratio = 0.75;
        let mut locked_vertices = HashSet::new();
        locked_vertices.insert(2);
        locked_vertices.insert(3);

        // Act
        let eligible_vertices_to_move = gain_conn_ratio_filter(
            &locked_vertices,
            &partitions,
            &gain,
            &vtx_conn_data_struct,
            filter_ratio);

        // Assert
        assert_eq!(eligible_vertices_to_move.len(), 2);
        assert_eq!(eligible_vertices_to_move[0], 0);
        assert_eq!(eligible_vertices_to_move[1], 1);
    }

    #[test]
    fn test_update_parts_and_vertex_connectivity(){
        // Arrange
        let mut adjacency = sprs::CsMat::empty(sprs::CSR, 0);
        adjacency.insert(0, 1, 1);
        adjacency.insert(0, 2, 2);
        adjacency.insert(2, 4, 3);
        adjacency.insert(4, 5, 1);
        adjacency.insert(5, 3, 3);
        adjacency.insert(3, 1, 2);
        adjacency.insert(1, 0, 1);
        adjacency.insert(2, 0, 2);
        adjacency.insert(4, 2, 3);
        adjacency.insert(5, 4, 1);
        adjacency.insert(3, 5, 3);
        adjacency.insert(1, 3, 2);

        let mut partitions = [0, 0, 0, 0, 1, 1];
        let mut vtx_conn_data_struct = init_vertex_connectivity_data_structure(
            &adjacency.view(),
            &partitions);
        let moves = vec![
            Move{
                vertex: 2,
                partition_id: 1,
            },
            Move{
                vertex: 3,
                partition_id: 1,
            }
        ];

        // Act
        update_parts_and_vertex_connectivity(&adjacency.view(),
                                             &mut partitions,
                                             &mut vtx_conn_data_struct,
                                             moves);

        // Assert
        assert_eq!(partitions[2], 1);
        assert_eq!(partitions[3], 1);
        assert_eq!(*vtx_conn_data_struct[0].get(&0).unwrap(), 1);
        assert_eq!(*vtx_conn_data_struct[0].get(&1).unwrap(), 2);
        assert_eq!(*vtx_conn_data_struct[1].get(&0).unwrap(), 1);
        assert_eq!(*vtx_conn_data_struct[1].get(&1).unwrap(), 2);
        assert_eq!(*vtx_conn_data_struct[4].get(&1).unwrap(), 4);
        assert_eq!(*vtx_conn_data_struct[5].get(&1).unwrap(), 4);
    }

    #[test]
    fn test_is_higher_placed(){
        // Arrange
        let gain = [4, 2, 2, 1];
        let list_of_vertices = [0, 1, 2];

        // Act
        let result1 = is_higher_placed(0, 2, &gain, &list_of_vertices);

        // Assert
        assert_eq!(result1, true);

        // Act
        let result2 = is_higher_placed(1, 2, &gain, &list_of_vertices);

        // Assert
        assert_eq!(result2, true);

        // Act
        let result3 = is_higher_placed(3, 2, &gain, &list_of_vertices);
        // Assert
        assert_eq!(result3, false);
    }

    #[test]
    fn test_get_weight_of_partition(){
        // Arrange
        let partitions = [1, 0, 0];
        let vertex_weights = [1f64, 2f64, 3f64];

        // Act
        let weight = get_weight_of_partition(0, &partitions, &vertex_weights);

        // Assert
        assert_eq!(weight, 5f64);
    }
    #[test]
    fn test_calculate_slot() {
        // Arrange and Act
        let slot1 = calculate_slot(-4, 3);
        let slot2 = calculate_slot(0, 3);
        let slot3 = calculate_slot(6, 8);
        let slot4 = calculate_slot(10, 3);

        // Assert
        assert_eq!(slot1, 0);
        assert_eq!(slot2, 1);
        assert_eq!(slot3, 4);
        assert_eq!(slot4, 3);
    }

    #[test]
    fn test_get_adjacent_eligible_destination_partitions(){
        // Arrange
        let mut adjacency = sprs::CsMat::empty(sprs::CSR, 0);
        adjacency.insert(0, 1, 1);
        adjacency.insert(0, 2, 2);
        adjacency.insert(0, 3, 3);
        adjacency.insert(2, 4, 3);
        adjacency.insert(1, 0, 1);
        adjacency.insert(2, 0, 2);
        adjacency.insert(3, 0, 3);
        adjacency.insert(4, 2, 3);

        let partitions = [0, 1, 3, 4, 2];
        let light_partitions = [1, 2];

        // Act
        let adjacent_eligible_partitions = get_adjacent_eligible_destination_partitions(
            &adjacency.view(),
            0,
            &partitions,
            &light_partitions);

        // Assert
        assert_eq!(adjacent_eligible_partitions.len(), 1);
        assert_eq!(adjacent_eligible_partitions[0], 1);
    }

    #[test]
    fn test_jetrw(){
        // Arrange
        let mut adjacency = sprs::CsMat::empty(sprs::CSR, 0);
        adjacency.insert(0, 1, 3);;
        adjacency.insert(1, 2, 3);
        adjacency.insert(2, 3, 3);
        adjacency.insert(3, 0, 3);
        adjacency.insert(1, 0, 3);;
        adjacency.insert(2, 1, 3);
        adjacency.insert(3, 2, 3);
        adjacency.insert(0, 3, 3);

        let vtx_weights = [300f64, 300f64, 300f64, 150f64];
        let partitions = [0, 0, 0, 1];

        // Act
        let vtx_conn_data_struct =
            init_vertex_connectivity_data_structure(
                &adjacency.view(),
                &partitions);
        let moves = jetrw(&adjacency.view(), &partitions, &vtx_weights, &vtx_conn_data_struct, 2, 0.1);

        // Assert
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].vertex, 0);
        assert_eq!(moves[0].partition_id, 1);
    }

    #[test]
    fn test_jetlp() {
        // Arrange
        let mut adjacency = sprs::CsMat::empty(sprs::CSR, 0);
        adjacency.insert(0, 1, 5);;
        adjacency.insert(1, 2, 8);
        adjacency.insert(2, 3, 1);
        adjacency.insert(3, 0, 2);
        adjacency.insert(1, 0, 5);;
        adjacency.insert(2, 1, 8);
        adjacency.insert(3, 2, 1);
        adjacency.insert(0, 3, 2);

        let partitions = [0, 1, 1, 0];
        let locked_vertices = HashSet::new();

        // Act
        let vtx_conn_data_struct = init_vertex_connectivity_data_structure(
            &adjacency.view(),
            &partitions);
        let moves = jetlp(&adjacency.view(),
                                      &partitions,
                                      &vtx_conn_data_struct,
                                      &locked_vertices,
                             0.3);

        // Assert
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].vertex, 0);
        assert_eq!(moves[0].partition_id, 1);
    }
}