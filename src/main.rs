//import time
use device_query::{DeviceQuery, DeviceState, Keycode};
use obj::{load_obj, Obj};
use std::fs::File;
use std::io::BufReader;
use std::io::{BufWriter, Write};
use std::time::Instant;
use termion;

struct FrameBuffer {
    front_buffer: Vec<u32>,
    back_buffer: Vec<u32>,
    width: usize,
    height: usize,
}

#[derive(Copy, Clone)]
struct Color {
    r: u8,
    g: u8,
    b: u8,
}

impl FrameBuffer {
    fn new(width: usize, height: usize, initial_color: Color) -> FrameBuffer {
        let initial_color_value = u32::from(initial_color.r) << 16
            | u32::from(initial_color.g) << 8
            | u32::from(initial_color.b);
        let framebuffer = FrameBuffer {
            front_buffer: vec![initial_color_value; width * height],
            back_buffer: vec![initial_color_value; width * height],
            width,
            height,
        };
        framebuffer.clear_terminal_and_fill_with_initial_color(initial_color);
        framebuffer
    }

    fn clear_terminal_and_fill_with_initial_color(&self, initial_color: Color) {
        let stdout = std::io::stdout();
        let mut out = BufWriter::new(stdout.lock());
        write!(out, "\x1B[2J\x1B[1;1H").unwrap();
        for _y in 0..self.height {
            for _x in 0..self.width {
                write!(
                    out,
                    "\x1b[48;2;{};{};{}m  ",
                    initial_color.r, initial_color.g, initial_color.b
                )
                .unwrap();
            }
            writeln!(out, "\x1b[0m").unwrap();
        }
        out.flush().unwrap();
    }
    fn set_pixel(&mut self, x: usize, y: usize, color: Color) {
        let color = u32::from(color.r) << 16 | u32::from(color.g) << 8 | u32::from(color.b);
        //check if x and y are in bounds
        if x >= self.width || y >= self.height {
            return;
        }
        self.back_buffer[y * self.width + x] = color;
    }

    fn get_pixel(&self, x: usize, y: usize) -> u32 {
        self.front_buffer[y * self.width + x]
    }

    fn draw_frame(&mut self) {
        let stdout = std::io::stdout();
        let mut out = BufWriter::new(stdout.lock());
        for y in 0..self.height {
            for x in 0..self.width {
                let front_pixel = self.get_pixel(x, y);
                let back_pixel = self.back_buffer[y * self.width + x];
                if front_pixel != back_pixel {
                    write!(
                        out,
                        "\x1B[{};{}H\x1b[48;2;{};{};{}m  ",
                        y + 1,
                        x * 2 + 1,
                        back_pixel >> 16,
                        (back_pixel >> 8) & 0xff,
                        back_pixel & 0xff
                    )
                    .unwrap();
                }
            }
        }
        write!(out, "\x1B[{};{}H\x1b[0m", self.height + 1, 1).unwrap();
        out.flush().unwrap();
        self.swap_buffers();
    }

    fn clear(&mut self) {
        self.back_buffer = vec![0; self.width * self.height];
    }

    fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.front_buffer, &mut self.back_buffer);
    }
}

fn draw_line(v1: UsizeVector2, v2: UsizeVector2, fb: &mut FrameBuffer) {
    // Draws a line from v1 to v2 using Bresenham's line algorithm
    let mut x1 = v1.x as i32;
    let mut y1 = v1.y as i32;
    let x2 = v2.x as i32;
    let y2 = v2.y as i32;

    let dx = (x2 - x1).abs();
    let dy = (y2 - y1).abs();

    let sx = if x1 < x2 { 1 } else { -1 };
    let sy = if y1 < y2 { 1 } else { -1 };

    let mut err = dx - dy;
    let mut e2;

    let white = Color {
        r: 255,
        g: 255,
        b: 255,
    };

    while x1 != x2 || y1 != y2 {
        fb.set_pixel(x1 as usize, y1 as usize, white);

        e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x1 += sx;
        }
        if e2 < dx {
            err += dx;
            y1 += sy;
        }
    }

    // Draw the last point
    fb.set_pixel(x2 as usize, y2 as usize, white);
}

fn fill_triangle(vv1: Vector2, vv2: Vector2, vv3: Vector2, fb: &mut FrameBuffer) {
    let hwid = fb.width as f32 / 2.;
    let hhei = fb.height as f32 / 2.;

    let mut v1: UsizeVector2 = UsizeVector2 {
        x: ((vv1.x + 1.) * hwid) as usize,
        y: ((vv1.y + 1.) * hhei) as usize,
        // y: ((-vv1.y + 1.) * hhei) as usize,
    };

    let mut v2: UsizeVector2 = UsizeVector2 {
        x: ((vv2.x + 1.) * hwid) as usize,
        y: ((vv2.y + 1.) * hhei) as usize,
        // y: ((-vv2.y + 1.) * hhei) as usize,
    };

    let mut v3: UsizeVector2 = UsizeVector2 {
        x: ((vv3.x + 1.) * hwid) as usize,
        y: ((vv3.y + 1.) * hhei) as usize,
        // y: ((-vv3.y + 1.) * hhei) as usize,
    };

    // Sort the vertices by y value
    if v1.y > v2.y {
        std::mem::swap(&mut v1, &mut v2);
    }
    if v1.y > v3.y {
        std::mem::swap(&mut v1, &mut v3);
    }
    if v2.y > v3.y {
        std::mem::swap(&mut v2, &mut v3);
    }

    // Calculate the slopes of the edges
    let mut s1 = (v2.x as f32 - v1.x as f32) / (v2.y as f32 - v1.y as f32);
    let s2 = (v3.x as f32 - v1.x as f32) / (v3.y as f32 - v1.y as f32);

    // Calculate the x values for the edges
    let mut x1 = v1.x as f32;
    let mut x2 = v1.x as f32;

    // Draw the top half of the triangle
    for y in v1.y..v2.y {
        draw_line(
            UsizeVector2::new(x1 as usize, y),
            UsizeVector2::new(x2 as usize, y),
            fb,
        );
        x1 += s1;
        x2 += s2;
    }

    // Calculate the slope of the bottom edge
    s1 = (v3.x as f32 - v2.x as f32) / (v3.y as f32 - v2.y as f32);

    // Calculate the x value for the left edge
    x1 = v2.x as f32;

    // Draw the bottom half of the triangle
    for y in v2.y..v3.y {
        draw_line(
            UsizeVector2::new(x1 as usize, y),
            UsizeVector2::new(x2 as usize, y),
            fb,
        );
        x1 += s1;
        x2 += s2;
    }
}

#[derive(Copy, Clone, Debug)]
struct Face {
    vertices: [Vector4; 3],
    transformation: Matrix44,
}

fn draw_faces(faces: &Vec<Face>, fb: &mut FrameBuffer, camera: &Camera) {
    let mut brok = false;
    for face in faces {
        let final_matrix = camera.get_pv_matrix().mul(face.transformation);
        let mut vertices: [Vector4; 3] = face.vertices;
        for vertex in vertices.iter_mut() {
            *vertex = transform_vertex(*vertex, final_matrix);
            if (vertex.x < -1.
                || vertex.x > 1.
                || vertex.y < -1.
                || vertex.y > 1.)
            {
                brok = true;
                break;
            }
        }
        if !brok {
            fill_triangle(
                Vector2::new(vertices[0].x, vertices[0].y),
                Vector2::new(vertices[1].x, vertices[1].y),
                Vector2::new(vertices[2].x, vertices[2].y),
                fb,
            );
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Vector2 {
    x: f32,
    y: f32,
}

impl Vector2 {
    fn new(x: f32, y: f32) -> Vector2 {
        Vector2 { x, y }
    }
}

#[derive(Copy, Clone, Debug)]
struct UsizeVector2 {
    x: usize,
    y: usize,
}

impl UsizeVector2 {
    fn new(x: usize, y: usize) -> UsizeVector2 {
        UsizeVector2 { x, y }
    }
}

#[derive(Copy, Clone, Debug)]
struct Vector3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vector3 {
    fn new(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3 { x, y, z }
    }

    fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    fn normalize(&self) -> Vector3 {
        let mag = self.magnitude();
        Vector3 {
            x: (self.x / mag),
            y: (self.y / mag),
            z: (self.z / mag),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Vector4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

impl Vector4 {
    fn new(x: f32, y: f32, z: f32, w: f32) -> Vector4 {
        Vector4 { x, y, z, w }
    }
}

#[derive(Copy, Clone, Debug)]
struct Matrix44 {
    m: [[f32; 4]; 4],
}

impl Matrix44 {
    fn identity() -> Matrix44 {
        Matrix44 {
            m: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1., 0.0, 0.0],
                [0.0, 0.0, 1., 0.0],
                [0.0, 0.0, 0.0, 1.],
            ],
        }
    }

    fn mul(&self, other: Matrix44) -> Matrix44 {
        let mut m = [[0.0; 4]; 4];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, elem) in row.iter_mut().enumerate().take(4) {
                *elem = self.m[i][0] * other.m[0][j]
                    + self.m[i][1] * other.m[1][j]
                    + self.m[i][2] * other.m[2][j]
                    + self.m[i][3] * other.m[3][j];
            }
        }
        Matrix44 { m }
    }

    //multiply a matrix by a vector

    fn mul_vec(&self, other: Vector4) -> Vector4 {
        let mut v = Vector4::new(0., 0., 0., 0.);
        v.x = self.m[0][0] * other.x
            + self.m[0][1] * other.y
            + self.m[0][2] * other.z
            + self.m[0][3] * other.w;
        v.y = self.m[1][0] * other.x
            + self.m[1][1] * other.y
            + self.m[1][2] * other.z
            + self.m[1][3] * other.w;
        v.z = self.m[2][0] * other.x
            + self.m[2][1] * other.y
            + self.m[2][2] * other.z
            + self.m[2][3] * other.w;
        v.w = self.m[3][0] * other.x
            + self.m[3][1] * other.y
            + self.m[3][2] * other.z
            + self.m[3][3] * other.w;
        v
    }

    fn translate(&mut self, x: f32, y: f32, z: f32) {
        let mut m = Matrix44::identity();
        m.m[0][3] = x;
        m.m[1][3] = y;
        m.m[2][3] = z;
        *self = self.mul(m);
    }

    fn rotate(&mut self, axis: Vector3, angle: f32) {
        let mut m = Matrix44::identity();

        m.m[0][0] = angle.cos() + axis.x * axis.x * (1. - angle.cos());
        m.m[0][1] = axis.x * axis.y * (1. - angle.cos()) - axis.z * angle.sin();
        m.m[0][2] = axis.x * axis.z * (1. - angle.cos()) + axis.y * angle.sin();

        m.m[1][0] = axis.y * axis.x * (1. - angle.cos()) + axis.z * angle.sin();
        m.m[1][1] = angle.cos() + axis.y * axis.y * (1. - angle.cos());
        m.m[1][2] = axis.y * axis.z * (1. - angle.cos()) - axis.x * angle.sin();

        m.m[2][0] = axis.z * axis.x * (1. - angle.cos()) - axis.y * angle.sin();
        m.m[2][1] = axis.z * axis.y * (1. - angle.cos()) + axis.x * angle.sin();
        m.m[2][2] = angle.cos() + axis.z * axis.z * (1. - angle.cos());

        *self = self.mul(m);
    }
}

#[derive(Copy, Clone, Debug)]
struct Camera {
    position: Vector3,
    rotation: Vector3,
    proj_matrix: Matrix44,
    view_matrix: Matrix44,
}

impl Camera {
    fn new(
        start_pos: Vector3,
        start_rotation: Vector3,
        fov: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Camera {
        let mut camera = Camera {
            position: start_pos,
            rotation: Vector3 {
                x: start_rotation.x.to_radians(),
                y: start_rotation.y.to_radians(),
                z: start_rotation.z.to_radians(),
            },
            proj_matrix: Matrix44::identity(),
            view_matrix: Matrix44::identity(),
        };
        camera.create_projection_matrix(fov, aspect, near, far);
        camera.calculate_view_matrix();
        camera
    }

    fn get_forward_vector(&self) -> Vector3 {
        Vector3 {
            x: self.rotation.y.sin() * self.rotation.x.cos(),
            y: self.rotation.x.sin(),
            z: -self.rotation.y.cos() * self.rotation.x.cos(),
        }
        .normalize()
    }

    fn move_forward(&mut self, distance: f32) {
        let forward = self.get_forward_vector();
        self.position.x += forward.x * distance;
        self.position.y += forward.y * distance;
        self.position.z += forward.z * distance;
        // self.calculate_view_matrix();
    }

    fn move_right(&mut self, distance: f32) {
        let forward = self.get_forward_vector();
        let right = Vector3::new(-forward.z, 0., forward.x).normalize();
        self.position.x += right.x * distance;
        self.position.y += right.y * distance;
        self.position.z += right.z * distance;
        // self.calculate_view_matrix();
    }

    fn move_up(&mut self, distance: f32) {
        self.position.y += distance;
        // self.calculate_view_matrix();
    }

    fn rotate_up(&mut self, angle: f32) {
        self.rotation.x += angle;
        // self.calculate_view_matrix();
    }

    fn rotate_right(&mut self, angle: f32) {
        self.rotation.y += angle;
        // self.calculate_view_matrix();
    }

    fn calculate_view_matrix(&mut self) {
        self.view_matrix = Matrix44::identity();
        self.view_matrix
            .rotate(Vector3::new(1., 0., 0.), self.rotation.x);
        self.view_matrix
            .rotate(Vector3::new(0., 1., 0.), self.rotation.y);
        self.view_matrix
            .rotate(Vector3::new(0., 0., 1.), self.rotation.z);
        self.view_matrix
            .translate(-self.position.x, -self.position.y, -self.position.z);
    }

    fn get_pv_matrix(&self) -> Matrix44 {
        self.proj_matrix.mul(self.view_matrix)
    }

    // fn create_projection_matrix(&mut self, fov: f32, aspect: f32, near: f32, far: f32) {
    //     let fov = fov.to_radians();
    //     let scale = 1.0 / (fov * 0.5).tan(); // Precompute to avoid duplicate calculation
    //     let fov_y = scale * aspect;
    //     let fov_x = scale;
    //     let f = far / (far - near);
    //     let nf = -(far * near) / (far - near);
    //     self.proj_matrix = Matrix44 {
    //         m: [
    //             [fov_x, 0.0, 0.0, 0.0],
    //             [0.0, fov_y, 0.0, 0.0],
    //             [0.0, 0.0, f, nf],
    //             [0.0, 0.0, 1.0, 0.0],
    //         ],
    //     };
    // }

    // fn create_projection_matrix(&mut self, fov: f32, aspect: f32, near: f32, far: f32) {
    //     let fov = fov.to_radians();
    //     let scale = 1.0 / (fov * 0.5).tan(); // Precompute to avoid duplicate calculation
    //     let fov_y = scale * aspect;
    //     let fov_x = scale;
    //     let f = far / (far - near);
    //     let nf = -(far * near) / (far - near);
    //     self.proj_matrix = Matrix44 {
    //         m: [
    //             [fov_x, 0.0, 0.0, 0.0],
    //             [0.0, fov_y, 0.0, 0.0],
    //             [0.0, 0.0, f, 1.0],  // Changed this line
    //             [0.0, 0.0, nf, 0.0], // Changed this line
    //         ],
    //     };
    // }

    fn create_projection_matrix(&mut self, fov: f32, aspect: f32, near: f32, far: f32) {
        let fov = fov.to_radians();
        // self.proj_matrix = Matrix44::identity();
        let scale = 1. / (fov / 2.).tan();

        // self.proj_matrix.m[0][0] = scale*aspect;
        // self.proj_matrix.m[1][1] = scale;
        // self.proj_matrix.m[2][2] = -far / (far-near);
        // self.proj_matrix.m[3][2] = -far * near / (far-near);
        // self.proj_matrix.m[2][3] = -1.;
        // self.proj_matrix.m[3][3] = 0.;

        self.proj_matrix = Matrix44 {
            m: [
                [scale, 0.0, 0.0, 0.0],
                [0.0, scale * aspect, 0.0, 0.0],
                [0.0, 0.0, -far / (far - near), -1.0],
                [0.0, 0.0, -far * near / (far - near), 0.0],
            ],
        }
    }
}

fn transform_vertex(vertex: Vector4, mv_matrix: Matrix44) -> Vector4 {
    let mut f: Vector4;

    f = mv_matrix.mul_vec(vertex);

    if (f.w != 1.) {
        f.x /= f.w;
        f.y /= f.w;
        f.z /= f.w;
        // f.w = 1.;
    }
    f
}

fn get_faces(input: BufReader<File>) -> Vec<Face> {
    let model: Obj = load_obj(input).unwrap();
    let mut faces: Vec<Face> = Vec::new();

    for i in 0..model.indices.len() / 3 {
        let face = Face {
            vertices: [
                Vector4::new(
                    model.vertices[model.indices[i * 3] as usize].position[0],
                    model.vertices[model.indices[i * 3] as usize].position[1],
                    model.vertices[model.indices[i * 3] as usize].position[2],
                    1.,
                ),
                Vector4::new(
                    model.vertices[model.indices[i * 3 + 1] as usize].position[0],
                    model.vertices[model.indices[i * 3 + 1] as usize].position[1],
                    model.vertices[model.indices[i * 3 + 1] as usize].position[2],
                    1.,
                ),
                Vector4::new(
                    model.vertices[model.indices[i * 3 + 2] as usize].position[0],
                    model.vertices[model.indices[i * 3 + 2] as usize].position[1],
                    model.vertices[model.indices[i * 3 + 2] as usize].position[2],
                    1.,
                ),
            ],
            transformation: Matrix44::identity(),
        };

        faces.push(face);
    }

    faces.sort_by(|a, b| {
        let a = a.vertices[0].z + a.vertices[1].z + a.vertices[2].z;
        let b = b.vertices[0].z + b.vertices[1].z + b.vertices[2].z;
        b.partial_cmp(&a).unwrap()
    });

    faces
}
fn main() {
    //get terminal size
    let terminal_size = termion::terminal_size().unwrap();
    let width = terminal_size.0 as usize;
    let height = terminal_size.1 as usize;

    let aspect_x = width / 2;
    let aspect_y = height;
    let rate = 1;

    // let aspect_x = 16;
    // let aspect_y = 9;
    // let rate = 10;
    let device_state = DeviceState::new();

    let _start_time = Instant::now();

    // let aspectX = 64;
    // let aspectY = 48;
    // let rate = 1;

    //load model
    let input = BufReader::new(File::open("./monke.obj").unwrap());
    
    let monke_faces = get_faces(input);

    let input = BufReader::new(File::open("./cube.obj").unwrap());
    let cube_faces = get_faces(input);

    let mut camera = Camera::new(
        Vector3 {
            x: 0.,
            y: 0.,
            z: 0.,
        },
        Vector3 {
            x: 0.,
            y: 0.,
            z: 0.,
        },
        90.,
        (aspect_x * rate) as f32 / (aspect_y * rate) as f32,
        1.,
        1000.,
    );

    let mut angle = 0.;

    let mut fb = FrameBuffer::new(aspect_x * rate, aspect_y * rate, Color { r: 0, g: 0, b: 0 });
    let fps = 64;

    loop {
        let keys: Vec<Keycode> = device_state.get_keys();
        for key in &keys {
            match key {
                Keycode::W => {
                    camera.move_forward(0.1);
                }
                Keycode::S => {
                    camera.move_forward(-0.1);
                }
                Keycode::A => {
                    camera.move_right(-0.1);
                }
                Keycode::D => {
                    camera.move_right(0.1);
                }
                Keycode::Q => {
                    camera.move_up(0.1);
                }
                Keycode::E => {
                    camera.move_up(-0.1);
                }
                Keycode::I => {
                    camera.rotate_up(0.05);
                }
                Keycode::K => {
                    camera.rotate_up(-0.05);
                }
                Keycode::J => {
                    camera.rotate_right(-0.05);
                }
                Keycode::L => {
                    camera.rotate_right(0.05);
                }
                _ => {}
            }
        }

        camera.calculate_view_matrix();

        let mut draw_list_faces: Vec<Face> = Vec::new();

        let mut monke_transformation = Matrix44::identity();

        let mut cube_transformation = Matrix44::identity();

        monke_transformation.translate(0.0, 0.0, -4.);

        cube_transformation.translate(0.0, 0.0, 4.);

        monke_transformation.rotate(
            Vector3 {
                x: 1.,
                y: 0.,
                z: 0.,
            },
            180_f32.to_radians(),
        );

        monke_transformation.rotate(
            Vector3 {
                x: 0.,
                y: 1.,
                z: 0.,
            },
            angle,
        );

        // angle += 0.01;

        let filtered_monke_faces =  monke_faces
            .iter()
            .filter(|face| {
                let mut in_frustum = true;
                for vertex in &face.vertices {
                    let mut v = vertex.clone();
                    v.w = 1.;
                    v = monke_transformation.mul_vec(v);
                    v = camera.view_matrix.mul_vec(v);
                    if v.z > v.w {
                        in_frustum = false;
                    }
                }
                in_frustum
            })

            //set transformation matrix
            .map(|face| {
                let mut face = face.clone();
                face.transformation = monke_transformation;
                face
            })

            .collect::<Vec<Face>>();

        let filtered_cube_faces =  cube_faces
            .iter()
            .filter(|face| {
                let mut in_frustum = true;
                for vertex in &face.vertices {
                    let mut v = vertex.clone();
                    v.w = 1.;
                    v = cube_transformation.mul_vec(v);
                    v = camera.view_matrix.mul_vec(v);
                    if v.z > v.w {
                        in_frustum = false;
                    }
                }
                in_frustum
            })

            //set transformation matrix

            .map(|face| {
                let mut face = face.clone();
                face.transformation = cube_transformation;
                face
            })

            .collect::<Vec<Face>>();

        //add monke to draw faces
        draw_list_faces.append(&mut filtered_monke_faces.clone());
        draw_list_faces.append(&mut filtered_cube_faces.clone());

        //TODO: remove faces that are not in front of the camera

        //remove faces that are not in frustum

        // draw_list_faces = draw_list_faces
        //     .into_iter()
        //     .filter(|face| {
        //         let mut in_frustum = true;
        //         for vertex in &face.vertices {
        //             let mut v = vertex.clone();
        //             v.w = 1.;
        //             v = monke_transformation.mul_vec(v);
        //             v = camera.view_matrix.mul_vec(v);
        //             if v.z > v.w {
        //                 in_frustum = false;
        //             }
        //         }
        //         in_frustum
        //     })
        //     .collect();



        fb.clear();
        draw_faces(&draw_list_faces, &mut fb, &camera);
        fb.draw_frame();

        std::thread::sleep(std::time::Duration::from_millis(1000 / fps));
    }
}
