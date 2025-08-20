use std::io::Write;
use forceatlas2;
use crate::{Error};
use crate::{Topology};
use num_traits::abs;
use rand::{thread_rng, Rng};
use faer::{prelude::*, Mat};
use rand::prelude::SliceRandom;
use approx::{assert_abs_diff_eq};
use faer::mat::AsMatRef;
use rand_distr::{Distribution, Normal};
use faer::matrix_free::LinOp;

fn geometric_partitioner<T>(
    partition: &mut [usize],
    weights: &[f64],
    adjacency: T,
    fa2_iterations: u64,
)
where
    T: Topology<i64> + Sync {
    let points_matrix = convert_graph_to_coordinates(&adjacency, weights.to_vec(), fa2_iterations);
    geopart(&adjacency, &points_matrix, 100, partition);
}

//This function projects d-dimensional points stereographically onto a (d+1)-dimensional unit sphere
fn stereo_up(xy: &Mat<f64>) -> Mat<f64>{
    let rows = xy.nrows();
    let dim = xy.ncols();
    let stereo_up_dim = dim + 1;
    let mut north_pole_vecs = Mat::<f64>::zeros(rows, stereo_up_dim);
    north_pole_vecs.col_as_slice_mut(stereo_up_dim - 1).fill(1.0);
    let mut squared_xy = Mat::<f64>::zeros(rows, dim);

    for r_idx in 0..rows {
        for c_idx in 0..dim {
            squared_xy[(r_idx, c_idx)] = xy[(r_idx, c_idx)]*xy[(r_idx, c_idx)];
        }
    }

    let mut norm_squared_xy = Mat::<f64>::zeros(rows, 1);

    for r_idx in 0..rows{
        norm_squared_xy[(r_idx, 0)] = squared_xy.row(r_idx).sum();
    }

    let norm_squared_denom = norm_squared_xy + Mat::from_fn(rows, 1, |_, _| 1.0);
    let mut xy_extended = Mat::<f64>::zeros(rows, stereo_up_dim);
    xy_extended.get_mut(0..rows,0..dim).copy_from(xy);
    xy_extended.col_as_slice_mut(dim).fill(-1.0);
    let mut xyz = Mat::<f64>::zeros(rows, stereo_up_dim);

    for r_idx in 0..rows {
        for c_idx in 0..stereo_up_dim {
            xyz[(r_idx, c_idx)] = north_pole_vecs[(r_idx, c_idx)] + 2.0 * xy_extended[(r_idx, c_idx)] / norm_squared_denom[(r_idx, 0)];
        }
    }

    xyz
}

// This function projects points from a (d+1)-dimensional unit sphere back to d-dimensional space
fn stereo_down(xyz: &Mat<f64>) -> Mat<f64> {
    let rows = xyz.nrows();
    let dim = xyz.ncols();
    let last_col = xyz.col(dim-1);
    let denom = Mat::from_fn(rows, 1, |_, _| 1.0).col(0) - last_col;
    let mut xy = Mat::<f64>::zeros(rows, dim - 1);

    for r_idx in 0..rows {
        for c_idx in 0..(dim - 1) {
            xy[(r_idx, c_idx)] = xyz[(r_idx, c_idx)] / denom[r_idx];
        }
    }

    xy
}

// This function computes a Householder reflection for a point.
fn reflector(c: &Mat<f64>) -> (Mat<f64>, f64) {
    let mut c_rev_transposed = Mat::<f64>::zeros(c.ncols(), c.nrows());

    for r_idx in 0..c.ncols() {
        c_rev_transposed[(r_idx, 0)] = c[(0, c.ncols() - r_idx - 1)];
    }

    let qr_decomp = c_rev_transposed.qr();
    let q = qr_decomp.compute_Q();
    let r = qr_decomp.R();

    let mut q_reordered = Mat::<f64>::zeros(c.ncols(), c.ncols());

    for r_idx in 0..c.ncols() {
        for c_idx in 0..c.ncols() {
            q_reordered[(r_idx, c_idx)] = q[(c.ncols() - 1 - r_idx, c.ncols() - 1 - c_idx)];
        }
    }

    (q_reordered, r[(0, 0)])
}

// This function applies a conformal map to points on a sphere, moving a specified centerpoint to
// the origin through a sequence of reflection, projection, and scaling operations.
fn con_map(c: &Mat<f64>, xyz: &Mat<f64>) -> (Mat<f64>, Mat<f64>) {
    let (q, r) = reflector(&c);
    let alpha = ((1.0 + r) / (1.0 - r)).sqrt();
    let xyz_ref = xyz * q;
    let xy_ref = stereo_down(&xyz_ref);
    let xy_map = xy_ref/alpha;
    let xyz_map = stereo_up(&xy_map);
    (xyz_map, xy_map)
}

// This function calculates a center point for a collection of data points using Radon Theorem.
fn centerpoint(xyz: &Mat<f64>, csample: usize) -> Mat<f64> {
    let num_points = xyz.nrows();
    let dim = xyz.ncols();
    let mut sample_size = csample.min(num_points);
    sample_size = ((dim as f64 + 1.0) * ((sample_size as f64 - 1.0) / (dim as f64 + 1.0).floor())) as usize + 1;
    let xyz_queue_capacity = (sample_size as f64 * (1.0 + 1.0 / dim as f64)).ceil() as usize;
    let mut xyzs = Mat::<f64>::zeros(xyz_queue_capacity, dim);
    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..num_points).collect();
    indices.shuffle(&mut rng);
    let sample_indices = &indices[0..sample_size];

    for (r_idx, &reordered_idx) in sample_indices.iter().enumerate() {
        xyzs.row_mut(r_idx).copy_from(&xyz.row(reordered_idx));
    }

    let mut queuehead = 0;
    let mut queuetail = sample_size-1;

    while queuehead < queuetail {
        queuetail += 1;
        let points_for_radon = xyzs.get(queuehead..queuehead+dim+2, 0..dim).to_owned();
        let radon_point = radon(&points_for_radon); //.to_owned() creates a DMatrix from the slice
        xyzs.row_mut(queuetail).copy_from(radon_point.row(0));
        queuehead += dim + 3;
    }

    let mut center_point = Mat::<f64>::zeros(1, dim);
    center_point.row_mut(0).copy_from(xyzs.row(queuetail));

    center_point
}

// This function computes the Radon point for a given set of points using Radon's theorem.
fn radon(points: &Mat<f64>) -> Mat<f64> {
    let (num_points, dim) = points.shape();

    if num_points != dim + 2 {
        panic!(
            "Radon function expects num_points ({}) = dim_plus_1 ({}) + 2",
            num_points, dim
        );
    }

    let mut augmented_matrix = Mat::<f64>::ones(num_points, dim+1);

    for r_idx in 0..num_points{
        for c_idx in 1..dim+1 {
            augmented_matrix[(r_idx, c_idx)] = points[(r_idx, c_idx-1)];
        }
    }

    let augmented_matrix = augmented_matrix.transpose().to_owned();
    let null_mat = null(&augmented_matrix);
    let null_mat_col0 = null_mat.col(0);
    let mut positive_coefficients_sum = 0.0;
    let mut positive_coefficients_points = Vec::new();
    let mut positive_coefficients = Vec::new();
    let mut num_rows = 0;

    for point_idx in 0..num_points {
        let coefficient = null_mat_col0[point_idx];

        if coefficient > 0.0 {
            num_rows += 1;
            positive_coefficients_sum += coefficient;
            let curr_point = points.row(point_idx).iter().cloned().collect::<Vec<f64>>();
            positive_coefficients_points.extend_from_slice(&curr_point);
            positive_coefficients.push(coefficient);

        }
    }

    let mut positive_coefficients_points_mat = Mat::<f64>::zeros(num_rows, dim);
    let mut positive_coefficients_mat = Mat::<f64>::zeros(1, num_rows);

    for r_idx in 0..num_rows{
        for c_idx in 0..dim {
            positive_coefficients_points_mat[(r_idx, c_idx)] = positive_coefficients_points[r_idx*dim + c_idx];
        }
    }

    for r_idx in 0..num_rows {
        positive_coefficients_mat[(0, r_idx)] = positive_coefficients[r_idx];
    }

    positive_coefficients_mat*positive_coefficients_points_mat / positive_coefficients_sum
}

// This function computes the null space of a matrix using Singular Value Decomposition (SVD).
fn null(matrix: &Mat<f64>) -> Mat<f64> {
    let cols = matrix.ncols();

    let svd = matrix.svd().unwrap();
    let v_t = svd.V().transpose();
    let s = svd.S();

    let mut rank = 0;

    for row in 0..s.nrows(){
        if s[row] > 0.0 {
            rank += 1;
        }
    }

    let mut null_mat = Mat::<f64>::zeros(cols,  cols - rank);

    for r_idx in rank..cols{
        let col = v_t.row(r_idx).transpose();

        for j in 0..col.nrows(){
            null_mat[(j, r_idx-rank)] = col[j]
        }
    }

    null_mat
}

// This function scales a set of points so they are centered at the origin
// and fit within a bounding box from -1 to 1 in each dimension.
fn scale_points(points: &Mat<f64>) -> Mat<f64>{
    let npoints = points.nrows();
    let dim = points.ncols();
    let mut xy_scaled = points.clone();
    let mut means = Mat::<f64>::zeros(1, dim);
    let mut max_abs_val = 0.0f64;

    for c_idx in 0..dim {
        let mean = xy_scaled.col(c_idx).sum()/dim as f64;
        means[(0, c_idx)] = mean;
    }

    xy_scaled = xy_scaled - Mat::<f64>::ones(npoints, 1)*means;

    for r_idx in 0..npoints {
        max_abs_val = abs(xy_scaled[(r_idx, 0)]).max(max_abs_val);
        max_abs_val = abs(xy_scaled[(r_idx, 1)]).max(max_abs_val);
    }

    if max_abs_val > 0.0 {
        xy_scaled /= max_abs_val;
    }

    xy_scaled
}

fn geopart<T>(graph: T, xy: &Mat<f64>, ntries: i64, partition: &mut [usize]) where T: Topology<i64> + Sync + Copy  {
    let npoints = xy.nrows();
    let dim = xy.ncols();
    let nlines = ((ntries as f64/2.)*(dim as f64/(dim as f64+1.))).floor();
    let nouter = ((ntries as f64 - nlines as f64 + 1.) as f64).log(20.0).ceil() as usize;
    let ninner = ((ntries as f64 - nlines)/nouter as f64).floor() as usize;

    let csample = npoints.min((dim + 3).pow(4));

    let xy_scaled = scale_points(&xy);

    let mut circle_quality = i64::MAX;
    let mut best_circle;
    let xyz = stereo_up(&xy_scaled);

    for _ in 0..nouter {
        let cpt = centerpoint(&xyz, csample);

        let (xyzmap, _) = con_map(&cpt, &xyz);
        let (great_circle, gc_quality) = sep_circle(graph, &xyzmap, ninner);

        if gc_quality < circle_quality {
            circle_quality = gc_quality;
            best_circle = great_circle.clone();
            perform_partition(&xyzmap, &best_circle, partition);
        }
    }
}

// This function finds the best separating great circle by generating random trials biased by inertial weighting
// to improve the quality of the resulting partition.
fn sep_circle<T>(graph: T, xyz: &Mat<f64>, ntries: usize) -> (Mat<f64>, i64) where T: Topology<i64> + Sync + Copy {
    let (_, dim) = xyz.shape();

    let xyz_transpose = xyz.transpose();
    let m = &xyz_transpose * xyz;
    let m_squared = &m * &m;

    let mut rng = thread_rng();
    let normal_dist = Normal::new(0.0, 1.0).unwrap();
    let random_normals = Mat::from_fn(ntries, dim, |_, _| normal_dist.sample(&mut rng));

    let sep_circles = &random_normals * &m_squared;

    let mut best_gc_quality = i64::MAX;
    let mut best_sep_circle = Mat::<f64>::new();

    for r_idx in 0..ntries {
        let circle = sep_circles.row(r_idx);
        let mut sep_circle = Mat::<f64>::zeros(1, dim);

        for c_idx in 0..dim {
            sep_circle[(0, c_idx)] = circle[c_idx];
        }

        let current_quality = sep_quality(&sep_circle, graph, xyz);

        if current_quality < best_gc_quality {
            best_gc_quality = current_quality;
            best_sep_circle = sep_circle;
        }
    }

    (best_sep_circle, best_gc_quality)
}

// This function calculates the quality of a partition defined by the edge-cut.
fn sep_quality<T>(v: &Mat<f64>, graph: T, xyz: &Mat<f64>) -> i64 where T: Topology<i64> + Sync{
    let mut partition = vec![0; graph.len()];
    perform_partition(xyz, &v, &mut partition);

    graph.edge_cut(&partition)
}
// This function computes the median of a vector of dot products, which serves as the threshold
// for dividing vertices into two balanced sets.
fn find_median(dot_prod_mat: &Mat<f64>) -> f64 {
    let mut dot_prod_vec = dot_prod_mat.col_as_slice(0).to_vec();
    dot_prod_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = dot_prod_vec.len();

    if len % 2 == 0 {
        let mid_right = len / 2;
        let mid_left = mid_right - 1;
        (dot_prod_vec[mid_left] + dot_prod_vec[mid_right]) / 2.0
    } else {
        dot_prod_vec[len / 2]
    }
}

// Partitions vertices into two partitions based on the separating plane.
fn perform_partition(xyz: &Mat<f64>, sep_plane: &Mat<f64>, partition: &mut [usize]) {
    let n = xyz.nrows();
    let dot_prod_mat = xyz*sep_plane.transpose();
    let median = find_median(&dot_prod_mat);

    let mut values_smaller_than_median_indices = Vec::new();
    let mut values_greater_than_median_indices= Vec::new();
    let mut values_equal_to_median_indices = Vec::new();

    for i in 0..n {
        let val = dot_prod_mat[(i, 0)];

        if val < median {
            values_smaller_than_median_indices.push(i);
        } else if val > median {
            values_greater_than_median_indices.push(i);
        } else {
            values_equal_to_median_indices.push(i);
        }
    }

    let num_of_values_equal_to_median = values_equal_to_median_indices.len();
    let num_of_values_less_than_median = values_smaller_than_median_indices.len();

    if num_of_values_equal_to_median > 0 {
        let num_of_values_for_balance_a = ((n as f64 / 2.0).ceil() as usize).saturating_sub(num_of_values_less_than_median);
        let num_of_values_to_assign_to_a = num_of_values_for_balance_a.min(num_of_values_equal_to_median);

        if num_of_values_to_assign_to_a > 0 {
            values_smaller_than_median_indices.extend_from_slice(&values_equal_to_median_indices[0..num_of_values_to_assign_to_a]);
        }

        if num_of_values_to_assign_to_a < num_of_values_equal_to_median {
            values_greater_than_median_indices.extend_from_slice(&values_equal_to_median_indices[num_of_values_to_assign_to_a..num_of_values_equal_to_median]);
        }
    }

    for i in values_smaller_than_median_indices{
        partition[i] = 1;
    }

    for i in values_greater_than_median_indices{
        partition[i] = 0;
    }
}

// This function computes co-ordinates for each node using forceatlas2 algorithm.
fn convert_graph_to_coordinates<T>(graph: T, weights: Vec<f64>, iter:u64) -> Mat<f64> where T: Topology<i64> + Sync{
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

    let num_points = layout.nodes.len();
    let mut points_mat = Mat::zeros(num_points, 2);

    for (i, node) in layout.nodes.iter().enumerate() {
        points_mat[(i, 0)] = node.pos.x();
        points_mat[(i, 1)] = node.pos.y();

    }

    points_mat
}

/// Geometric Partitioner
///
/// An implementation of the Geometric Partitioner algorithm
/// for graph partition.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), coupe::Error> {
/// use std::path::Path;
/// use rand::{thread_rng, Rng};
/// use sprs::{io, CsMat, TriMat};
/// use coupe::{GeometricPartitioner, Partition as _, Partition, Topology};
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
/// GeometricPartitioner {..Default::default()}.partition(&mut partition, (graph.view(), &weights))?;
/// let edge_cut = graph.view().edge_cut(&partition);
///
/// // Note: The edge cut is not theoretically guaranteed to lie between 700,000,000 and 800,000,000.
/// // However, experiments consistently produced values within this range, so the following assertion
/// // is used as a practical check.
///
/// assert!(edge_cut >= 700000000 && edge_cut <= 800000000);
/// Ok(())
/// }
/// ```
///
/// # Reference
///
/// Gilbert, John R., Gary L. Miller, and Shang-Hua Teng.
/// "Geometric mesh partitioning: Implementation and experiments."
/// SIAM Journal on Scientific Computing 19, no. 6 (1998): 2091-2110.

#[derive(Debug, Clone, Copy)]
pub struct GeometricPartitioner {
    pub fa2_iterations: u64,
}
impl Default for GeometricPartitioner {
    fn default() -> Self {
        GeometricPartitioner {
            fa2_iterations: 100,
        }
    }
}
impl<'a, T> crate::Partition<(T, &'a [f64])> for GeometricPartitioner
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
        let metadata = geometric_partitioner(
            part_ids,
            weights,
            adjacency,
            self.fa2_iterations
        );
        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use faer::{mat};
    use sprs::CsMat;
    use crate::Partition;
    use super::*;

    fn assert_matrices_approx_eq(left_mat: &Mat<f64>, right_mat: &Mat<f64>, tolerance: f64) {
        if left_mat.nrows() != right_mat.nrows() || left_mat.ncols() != right_mat.ncols() {
            panic!("Matrices have different dimensions: a is {}x{}, b is {}x{}", left_mat.nrows(), left_mat.ncols(), right_mat.nrows(), right_mat.ncols());
        }

        for i in 0..left_mat.nrows() {
            for j in 0..left_mat.ncols() {
                let diff = (left_mat[(i, j)] - right_mat[(i, j)]).abs();
                if diff > tolerance {
                    panic!(
                        "Matrices differ at element ({}, {}). Left Matrix: {}, Right Matrix: {}, Diff: {}, Tolerance: {}",
                        i,
                        j,
                        left_mat[(i, j)],
                        right_mat[(i, j)],
                        diff,
                        tolerance
                    );
                }
            }
        }
    }

    #[test]
    fn test_stereo_up() {
        // Arrange
        let mat = mat![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

        // Act
        let stereo_up_mat = stereo_up(&mat);

        // Assert
        let expected_stereo_up_mat = mat![[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert_matrices_approx_eq(&stereo_up_mat, &expected_stereo_up_mat, 1e-9);
    }

    #[test]
    fn test_stereo_down() {
        // Arrange
        let mat = mat![[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        // Act
        let stereo_down_mat = stereo_down(&mat);

        // Assert
        let expected_stereo_down_mat = mat![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        assert_matrices_approx_eq(&stereo_down_mat, &expected_stereo_down_mat, 1e-9);
    }

    #[test]
    fn test_reflector() {
        // Arrange
        let centerpoint = mat![[3.0, 0.0, 0.0]];

        // Act
        let (q_mat, r) = reflector(&centerpoint);

        // Assert
        let expected_q_mat = mat![[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]];
        assert_matrices_approx_eq(&q_mat, &expected_q_mat, 1e-9);
        assert_abs_diff_eq!(r, -3.0, epsilon=1e-9);
    }

    #[test]
    fn test_null() {
        // Arrange
        let mat = mat![[1.0, 1.0, 1.0, 1.0, 1.0],
                                 [1.0, 4.0, 7.0, 10.0, 13.0],
                                 [2.0, 5.0, 8.0, 11.0, 14.0],
                                 [3.0, 6.0, 9.0, 12.0, 15.0]];

        // Act
        let null_space = null(&mat);

        // Assert
        let zero_mat = &mat*&null_space;
        assert_matrices_approx_eq(&zero_mat, &mat![[0.0], [0.0], [0.0], [0.0]], 1e-9);
    }

    #[test]
    fn test_radon(){
        // Arrange
        let mat = mat![[-1.,0.], [1.,0.], [0.,1.], [0.,-1.]];

        // Act
        let radon_point = radon(&mat);

        // Assert
        let expected_radon_point = mat![[0.0, 0.0]];
        assert_matrices_approx_eq(&radon_point, &expected_radon_point, 1e-9);
    }

    #[test]
    fn test_centerpoint(){
        // Arrange
        let co_ordinates = mat![[1.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0],
                                          [-1.0, 0.0, 0.0],
                                          [0.0, 0.0, 1.0],
                                          [0.0, 0.0, -1.0]];

        // Act
        let n = 5;
        let calculated_cp = centerpoint(&co_ordinates, n);

        // Assert
        assert_matrices_approx_eq(&calculated_cp, &mat![[0.0, 0.0, 0.0]], 1e-9);
    }

    #[test]
    fn test_find_median(){
        // Arrange
        let data_odd_points = mat![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let data_even_points = mat![[1.0], [2.0], [3.0], [4.0]];

        // Act
        let median_even_points = find_median(&data_even_points);
        let median_odd_points  = find_median(&data_odd_points);

        // Assert
        assert_abs_diff_eq!(median_odd_points, 3.0, epsilon=1e-9);
        assert_abs_diff_eq!(find_median(&data_even_points), 2.5, epsilon=1e-9);
    }

    #[test]
    fn test_perform_partition(){
        // Arrange
        let sep_plane = mat![[1.0, 0.0, 1.0]];
        let points = mat![[3.0, 0.0, -3.0], [2.0, 0.0, -2.0], [-4.0, 0.0, 3.0], [-3.0, 0.0, 4.0]];
        let mut partition = [0; 4];

        // Act
        perform_partition(&points, &sep_plane, &mut partition);

        // Assert
        assert_eq!(partition, [1, 0, 1, 0]);
    }

    #[test]
    fn test_con_map(){
        // Arrange
        let cp = mat![[0.5, 0.5, 0.2]];
        let xyz = mat![[1.0, 1.0, 1.0],
                                 [0.0, 1.0, 0.0],
                                 [1.0, 0.0, 0.0],
                                 [1.0, 1.0, -1.0],
                                 [2.0, 1.0, 0.0]];

        // Act
        let(xyz_map, _) = con_map(&cp, &xyz);

        // Assert
        let expected_xyz_map = mat![[-0.6033882274525963, -0.6033882274525963, -0.5213878536590852],
                                              [-0.49364135691434713, 0.8628246397107064, 0.10886621079036374],
                                              [0.8628246397107064, -0.49364135691434713, 0.10886621079036374],
                                              [0.626886756757716, 0.626886756757716, 0.46262942881272107],
                                              [0.9611834276529786, -0.09709924432659642, -0.2582598597468746]];


        assert_matrices_approx_eq(&xyz_map, &expected_xyz_map, 1e-9);

    }

    #[test]
    fn test_scale_points() {
        // Arrange
        let xy = mat![[5.0, 10.0], [3.0, 7.0], [9.0, 5.0]];

        // Act
        let xy_scaled = scale_points(&xy);

        // Assert
        let expected_xy_scaled = mat![[-0.5833333333333333, -0.16666666666666666],
                                                [-0.9166666666666666, -0.6666666666666666],
                                                [0.08333333333333333, -1.0]];
        assert_matrices_approx_eq(&xy_scaled, &expected_xy_scaled, 1e-9);
    }
}