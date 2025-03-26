{{ bindings_code }}


const N_VERTICES: u32 = {{ n_vertices }};
const N_VERTICES_PER_VOXEL: u32 = {{ vertices_per_voxel }};
const N_COEFFICIENTS: u32 = {{ n_coefficients }};
const N_INDICES: u32 = {{ n_indices }};

fn get_voxel_id(id:u32) -> u32 {
  return u32(id / N_VERTICES_PER_VOXEL);
}

fn evaluate_sf_func(voxel_id: u32, vertex_id: u32) -> f32 {
  var sf_eval: f32 = 0.0;
  for (var i: u32 = 0; i < N_COEFFICIENTS; i += 1u) {

    sf_eval += load_s_sh_coef(i32(voxel_id * N_COEFFICIENTS + i)) * load_s_sf_func(i32(vertex_id * N_COEFFICIENTS + i));
  }
  return sf_eval;
}

fn update_normal(id: u32) {
  var accumulated: vec3<f32> = vec3<f32>(0.0);
  for (var i: u32 = 0; i < N_INDICES; i += 1u) {
    let index: vec3<i32> = load_s_indices(i32(i));
    let position0: vec3<f32> = load_s_positions(index.x);
    let position1: vec3<f32> = load_s_positions(index.y);
    let position2: vec3<f32> = load_s_positions(index.z);

    let a = s_radii[index.x + i32(id)] * position0;
    let b = s_radii[index.y + i32(id)] * position1;
    let c = s_radii[index.z + i32(id)] * position2;
    accumulated += normalize(cross(b - a, c - a));
  }
  s_normals[id * 3] = accumulated.x;
  s_normals[id * 3 + 1] = accumulated.y;
  s_normals[id * 3 + 2] = accumulated.z;
}

@compute @workgroup_size(128, 128, 128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let id:u32 = global_id.x;
  let vertex_id:u32 = id % N_VERTICES_PER_VOXEL;

  let voxel_id = get_voxel_id(id);
  s_radii[id] = evaluate_sf_func(voxel_id, vertex_id);

  update_normal(id);
}