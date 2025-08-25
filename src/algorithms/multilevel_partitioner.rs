use std::collections::{HashMap, HashSet};
use rand::seq::SliceRandom;
use rand::{SeedableRng};
use rand::rngs::StdRng;
use sprs::{CsMat, CsMatView};
use crate::{Error, Point2D, Rcb, Topology};
use crate::{JetPartitioner, Partition, GeometricPartitioner};
use crate::imbalance::imbalance;

#[derive(Clone, Copy, Debug)]
enum InitialPartitioner {
    GeometricPartitioner,
    RecursiveCoordinateBisection
}

fn multilevel_partitioner(
    partition: &mut [usize],
    weights: &[f64],
    adjacency: CsMat<i64>,
    initial_partitioner: InitialPartitioner,
    tolerance: f64
) {
    let mut coarse_graph_after_operation = adjacency.to_owned();
    let mut coarse_graphs = Vec::new();
    let mut vertex_mappings = Vec::new();
    let mut weights_coarse_graphs = Vec::new();

    let mut weights_of_coarse_graph_after_operation = weights.to_vec();

    while coarse_graph_after_operation.view().len() > 100  {
        let (coarse_graph, vertex_mapping, weights_of_coarse_graph) = heavy_edge_matching_coarse(coarse_graph_after_operation.view(), None, &weights_of_coarse_graph_after_operation);
        coarse_graph_after_operation = coarse_graph.to_owned();
        weights_of_coarse_graph_after_operation = weights_of_coarse_graph.clone();
        coarse_graphs.push(coarse_graph);
        vertex_mappings.push(vertex_mapping);
        weights_coarse_graphs.push(weights_of_coarse_graph);
    }

    let mut coarse_graph_partition = vec![0; coarse_graph_after_operation.view().len()];
    match initial_partitioner {
        InitialPartitioner::GeometricPartitioner => {
            GeometricPartitioner {..Default::default()}.partition(&mut coarse_graph_partition, (coarse_graph_after_operation.view(), &weights_of_coarse_graph_after_operation)).unwrap();
        },
        InitialPartitioner::RecursiveCoordinateBisection => {
            let points = convert_graph_to_coordinates(&coarse_graph_after_operation.view(), weights_of_coarse_graph_after_operation.clone(), 100);
            Rcb { iter_count: 1, ..Default::default()}.partition(&mut coarse_graph_partition, (points, weights_of_coarse_graph_after_operation.clone())).unwrap();
        }

    }

    let mut index = coarse_graphs.len() - 2;

    while index >= 0 {
        coarse_graph_partition = partition_uncoarse(&coarse_graph_partition, &vertex_mappings[index+1]);
        JetPartitioner { tolerance_factor: tolerance, ..Default::default()}.partition(&mut coarse_graph_partition, (coarse_graphs[index].view(), &weights_coarse_graphs[index])).unwrap();

        if index == 0 {
            break;
        }
        index -= 1;
    }
    let final_graph_partition = partition_uncoarse(&coarse_graph_partition, &vertex_mappings[0]);
    partition.copy_from_slice(&final_graph_partition);
}

// This function coarsens the graph using heavy edge matching algorithm.
fn heavy_edge_matching_coarse<T>(graph: T, seed: Option<u64>, weights: &[f64]) -> (CsMat<i64>, Vec<Vec<usize>>, Vec<f64>) where
    T: Topology<i64> + Sync{

    let mut rng = match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy()
    };

    let mut matched_nodes: HashSet<usize> = HashSet::new();
    let mut vertex_mapping = Vec::new();
    let mut old_vertex_to_new_vertex =  HashMap::new();
    let mut new_coarse_graph  = CsMat::empty(sprs::CSR, 0);

    let mut vertices: Vec<usize> = (0..graph.len()).collect();
    vertices.shuffle(&mut rng);

    let mut super_vertice: usize = 0;
    for vertice in vertices{
        if matched_nodes.contains(&vertice){
            continue;
        }

        let mut heaviest_edge_weight = 0;
        let mut heaviest_edge_connected_vertice = None;

        for (neighbor_vertex, edge_weight) in graph.neighbors(vertice){
            if edge_weight > heaviest_edge_weight && !matched_nodes.contains(&neighbor_vertex) {
                heaviest_edge_weight = edge_weight;
                heaviest_edge_connected_vertice = Some(neighbor_vertex);
            }
        }

        if !heaviest_edge_connected_vertice.is_none() {
            vertex_mapping.push(vec![vertice.min(heaviest_edge_connected_vertice.unwrap()),
                                           vertice.max(heaviest_edge_connected_vertice.unwrap())]);

            matched_nodes.insert(vertice);
            matched_nodes.insert(heaviest_edge_connected_vertice.unwrap());
            old_vertex_to_new_vertex.insert(vertice, super_vertice);
            old_vertex_to_new_vertex.insert(heaviest_edge_connected_vertice.unwrap(), super_vertice);
        } else {
            vertex_mapping.push(vec![vertice]);
            matched_nodes.insert(vertice);
            old_vertex_to_new_vertex.insert(vertice, super_vertice);
        }
        super_vertice += 1;
    }

    let mut edge_to_weight_mapping = HashMap::new();

    for vertex in 0..graph.len() {
        for (neighbor, edge_weight) in graph.neighbors(vertex){
            if old_vertex_to_new_vertex[&vertex] != old_vertex_to_new_vertex[&neighbor] {
                let key = old_vertex_to_new_vertex[&vertex].to_string() + "_" + &old_vertex_to_new_vertex[&neighbor].to_string();
                let total_edge_weight = edge_to_weight_mapping.entry(key).or_insert(0);
                *total_edge_weight += edge_weight;
            }
        }
    }

    for key in edge_to_weight_mapping.keys(){
        let parts: Vec<&str> = key.split('_').collect();
        let vertice1: usize = parts[0].parse().unwrap();
        let vertice2: usize = parts[1].parse().unwrap();
        let edge_weight = edge_to_weight_mapping.get(key).unwrap();

        new_coarse_graph.insert(vertice1, vertice2, *edge_weight);
    }

    let mut weights_coarse_graph = vec![0f64; new_coarse_graph.view().len()];

    for coarse_vertex in 0..vertex_mapping.len(){
        for uncoarse_vertex in vertex_mapping[coarse_vertex].iter(){
            weights_coarse_graph[coarse_vertex] += weights[*uncoarse_vertex];
        }
    }

    (new_coarse_graph, vertex_mapping, weights_coarse_graph)
}

// Refines the partition from a coarse graph back to the original finer graph.
fn partition_uncoarse(partition: &[usize], vertex_mapping: &Vec<Vec<usize>>) -> Vec<usize>{
    let mut vertices = 0;
    for mapped_vertices in vertex_mapping{
        vertices += mapped_vertices.len();
    }

    let mut new_partition: Vec<usize> = vec![0; vertices];

    for coarse_graph_vertice in 0..vertex_mapping.len(){
        let vertex_partition = partition[coarse_graph_vertice];

        for uncoarse_graph_vertice in &vertex_mapping[coarse_graph_vertice]{
            new_partition[*uncoarse_graph_vertice] = vertex_partition;
        }
    }

    new_partition
}

// This function computes 2D coordinates for graph nodes using forceatlas2 algorithm.
fn convert_graph_to_coordinates<T>(graph: &T, weights: Vec<f64>, iter:u64) -> Vec<Point2D> where T: Topology<i64> + Sync{
    let mut edges = Vec::new();

    for node in 0..graph.len() {
        for (neighbor_node, edge_weight) in graph.neighbors(node) {
            edges.push(((node, neighbor_node), edge_weight as f64));
        }
    }

    let mut layout = forceatlas2::Layout::<f64, 2>::from_graph_with_degree_mass(
        edges,
        weights,
        forceatlas2::Settings{strong_gravity: true , ..Default::default()},
    );

    for _ in 0..iter {
        layout.iteration();
    }

    let mut points = Vec::new();
    for (i, node) in layout.nodes.iter().enumerate() {
        points.push(Point2D::new(node.pos.x(), node.pos.y()));
    }

    points
}
/// Geometric Partitioner
///
/// An implementation of the Multilevel (Heavy Edge Matching && (Recursive Coordinate Bisection || Geometric Partitioner))
/// Partitioner algorithm for graph partition.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), coupe::Error> {
/// use std::path::Path;
/// use rand::{thread_rng, Rng};
/// use sprs::{io, CsMat, TriMat};
/// use coupe::{MultiLevelPartitioner, Topology, Partition as _};
/// use coupe::imbalance::imbalance;
/// use coupe::Point2D;
/// let vt2010_file_path = Path::new("vt2010.mtx");
/// let tri_mat: TriMat<i64> = io::read_matrix_market(vt2010_file_path).unwrap();
/// let mut graph: CsMat<i64> = tri_mat.to_csr();
/// let mut rng = thread_rng();
/// let weights: Vec<f64> = (0..graph.view().len())
///         .map(|_| rng.gen_range(1..100) as f64)
///         .collect();
///
/// let mut partition = vec![0; graph.view().len()];
///
/// MultiLevelPartitioner {..Default::default()}.partition(&mut partition, (graph.view(), &weights))?;
/// let edge_cut = graph.view().edge_cut(&partition);
///
/// // Note: The edge cut is not theoretically guaranteed to lie between 20,000,000 and 30,000,000.
/// // However, experiments consistently produced values within this range, so the following assertion
/// // is used as a practical check.
///
/// assert!(edge_cut >= 20000000 && edge_cut <= 30000000);
/// Ok(())
/// }
/// ```
///

#[derive(Debug, Clone, Copy)]
pub struct MultiLevelPartitioner {
    pub initial_partitioner: InitialPartitioner,
    pub tolerance: f64,
}

impl Default for MultiLevelPartitioner {
    fn default() -> Self {
        MultiLevelPartitioner {
            initial_partitioner: InitialPartitioner::RecursiveCoordinateBisection,
            tolerance: 0.1,
        }
    }
}

impl<'a> Partition<(CsMatView<'_, i64>, &'a [f64])> for MultiLevelPartitioner
{
    type Metadata = ();
    type Error = Error;

    fn partition(
        &mut self,
        part_ids: &mut [usize],
        (adjacency, weights): (CsMatView<i64>, &'a [f64]),
    ) -> Result<Self::Metadata, Self::Error> {

        if part_ids.len() != weights.len() {
            return Err(Error::InputLenMismatch {
                expected: part_ids.len(),
                actual: weights.len(),
            });
        }
        if part_ids.len() != adjacency.view().len() {
            return Err(Error::InputLenMismatch {
                expected: part_ids.len(),
                actual: adjacency.view().len(),
            });
        }
        let metadata = multilevel_partitioner(
            part_ids,
            weights,
            adjacency.to_owned(),
            self.initial_partitioner,
            self.tolerance
        );
        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use crate::imbalance::imbalance;
    use super::*;

    #[test]
    fn test_3_node_heavy_edge_matching_coarse() {
        // Arrange
        let mut graph = CsMat::empty(sprs::CSR, 0);
        graph.insert(0, 1, 5);
        graph.insert(0, 2, 10);
        graph.insert(1, 2, 15);

        graph.insert(1, 0, 5);
        graph.insert(2, 0, 10);
        graph.insert(2, 1, 15);

        let weights = [3.0, 4.0, 5.0];
        let seed = Some(5);

        // Act
        let (coarse_graph, vertex_mapping, weights_coarse_graph) = heavy_edge_matching_coarse(&graph.view(), seed, &weights);


        // Assert
        assert_eq!(15, *coarse_graph.get(0, 1).unwrap());
        assert_eq!(15, *coarse_graph.get(1, 0).unwrap());

        assert!(coarse_graph.get(0, 0).is_none());
        assert!(coarse_graph.get(1, 1).is_none());

        assert_eq!(vertex_mapping[0], vec![1, 2]);
        assert_eq!(vertex_mapping[1], vec![0]);

        assert_eq!(weights_coarse_graph, vec![9.0, 3.0]);
    }

    #[test]
    fn test_5_node_heavy_edge_matching_coarse() {
        // Arrange
        let mut graph = CsMat::empty(sprs::CSR, 0);
        graph.insert(0, 1, 3);
        graph.insert(1, 2, 5);
        graph.insert(2, 3, 4);
        graph.insert(3, 4, 6);
        graph.insert(4, 0, 10);

        graph.insert(1, 0, 3);
        graph.insert(2, 1, 5);
        graph.insert(3, 2, 4);
        graph.insert(4, 3, 6);
        graph.insert(0, 4, 10);

        let seed = Some(5);

        let weights = [1.0, 2.0, 3.0, 4.0, 5.0];

        // Act
        let (coarse_graph, vertex_mapping, weights_coarse_graph) = heavy_edge_matching_coarse(&graph.view(), seed, &weights);

        // Assert
        assert_eq!(6, *coarse_graph.get(0, 1).unwrap());
        assert_eq!(6, *coarse_graph.get(1, 0).unwrap());

        assert_eq!(3, *coarse_graph.get(0, 2).unwrap());
        assert_eq!(3, *coarse_graph.get(2, 0).unwrap());

        assert_eq!(5, *coarse_graph.get(1, 2).unwrap());
        assert_eq!(5, *coarse_graph.get(2, 1).unwrap());

        assert!(coarse_graph.get(0, 0).is_none());
        assert!(coarse_graph.get(1, 1).is_none());
        assert!(coarse_graph.get(2, 2).is_none());

        assert_eq!(vertex_mapping[0], vec![0, 4]);
        assert_eq!(vertex_mapping[1], vec![2, 3]);
        assert_eq!(vertex_mapping[2], vec![1]);

        assert_eq!(weights_coarse_graph, vec![6.0, 7.0, 2.0]);
    }

    #[test]
    fn test_partition_uncoarse() {
        // Arrange
        let vertex_mapping = vec![vec![0, 3], vec![2], vec![1]];
        let weights_coarse_graph = [5.0, 7.0, 6.0];
        let coarse_graph_partition = [1, 0, 0];
        let weights_uncoarse_graph = [2.0, 6.0, 7.0, 3.0];

        // Act
        let uncoarsed_graph_partition = partition_uncoarse(&coarse_graph_partition, &vertex_mapping);

        // Assert
        assert_eq!(uncoarsed_graph_partition, vec![1, 0, 0, 1]);
        let epsilon = 1e9;
        let coarse_graph_imbalance = imbalance(2, &coarse_graph_partition, weights_coarse_graph.clone());
        let uncoarse_graph_imbalance = imbalance(2, &uncoarsed_graph_partition, weights_uncoarse_graph.clone());
        assert!((coarse_graph_imbalance - uncoarse_graph_imbalance).abs() < epsilon);
    }
}