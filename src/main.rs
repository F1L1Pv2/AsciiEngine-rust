//import time
use device_query::{DeviceQuery, DeviceState, Keycode};
use obj::{load_obj, Obj};
use std::cmp;
use std::fs::File;
use std::io::BufReader;
use std::io::{BufRead, BufWriter, Write};
use std::time::{Duration, Instant};
use std::{thread, time};

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
        let initial_color_value =
            (initial_color.r as u32) << 16 | (initial_color.g as u32) << 8 | initial_color.b as u32;
        let mut framebuffer = FrameBuffer {
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
        for y in 0..self.height {
            for x in 0..self.width {
                write!(
                    out,
                    "\x1b[48;2;{};{};{}m  ",
                    initial_color.r, initial_color.g, initial_color.b
                )
                .unwrap();
            }
            write!(out, "\x1b[0m\n").unwrap();
        }
        out.flush().unwrap();
    }
    fn set_pixel(&mut self, x: usize, y: usize, color: Color) {
        let color = (color.r as u32) << 16 | (color.g as u32) << 8 | color.b as u32;
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

    let mut v1: UsizeVector2 = UsizeVector2{ 
        x:((vv1.x + 1.) * hwid) as usize,
        // y: ((vv1.y + 1.) * hhei) as usize,
        y:((-vv1.y + 1.) * hhei) as usize,
    };

    let mut v2: UsizeVector2 = UsizeVector2{ 
        x:((vv2.x + 1.) * hwid) as usize,
        // y: ((vv2.y + 1.) * hhei) as usize,
        y:((-vv2.y + 1.) * hhei) as usize,
    };

    let mut v3: UsizeVector2 = UsizeVector2{ 
        x:((vv3.x + 1.) * hwid) as usize,
        // y: ((vv3.y + 1.) * hhei) as usize,
        y:((-vv3.y + 1.) * hhei) as usize,
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
    let mut s2 = (v3.x as f32 - v1.x as f32) / (v3.y as f32 - v1.y as f32);

    // Calculate the x values for the edges
    let mut x1 = v1.x as f32;
    let mut x2 = v1.x as f32;

    // Draw the top half of the triangle
    for y in v1.y..v2.y {
        draw_line(UsizeVector2::new(x1 as usize, y), UsizeVector2::new(x2 as usize, y), fb);
        x1 += s1;
        x2 += s2;
    }

    // Calculate the slope of the bottom edge
    s1 = (v3.x as f32 - v2.x as f32) / (v3.y as f32 - v2.y as f32);

    // Calculate the x value for the left edge
    x1 = v2.x as f32;

    // Draw the bottom half of the triangle
    for y in v2.y..v3.y {
        draw_line(UsizeVector2::new(x1 as usize, y), UsizeVector2::new(x2 as usize, y), fb);
        x1 += s1;
        x2 += s2;
    }
}

#[derive(Copy, Clone, Debug)]
struct Face {
    vertices: [Vector4; 3],
    color: Vector3,
}

impl Face {
    fn new(vertices: [Vector4; 3], color: Vector3) -> Face {
        Face { vertices, color }
    }
}

fn drawFaces(faces: &Vec<Face>, transformation: Matrix44, fb: &mut FrameBuffer, Camera: &Camera) {
    let finalMatrix = Camera.getPvMatrix().mul(transformation);
    for face in faces {
        let mut vertices: [Vector4; 3] = face.vertices;
        for i in 0..3 {
            vertices[i] = transformVertex(vertices[i], finalMatrix);
        }
        fill_triangle(
            Vector2::new(vertices[0].x, vertices[0].y),
            Vector2::new(vertices[1].x, vertices[1].y),
            Vector2::new(vertices[2].x, vertices[2].y),
            fb,
        );
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

    fn zero() -> Vector2 {
        Vector2 { x: 0., y: 0. }
    }

    fn one() -> Vector2 {
        Vector2 { x: 1., y: 1. }
    }

    fn cross(&self, other: Vector2) -> f32 {
        self.x * other.y - self.y * other.x
    }

    fn add(&self, other: Vector2) -> Vector2 {
        Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    fn sub(&self, other: Vector2) -> Vector2 {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    fn mul(&self, other: Vector2) -> Vector2 {
        Vector2 {
            x: self.x * other.x,
            y: self.y * other.y,
        }
    }

    fn dot(&self, other: Vector2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    fn normalize(&self) -> Vector2 {
        let mag = self.magnitude();
        Vector2 {
            x: self.x / mag,
            y: self.y / mag,
        }
    }

    fn distance(&self, other: Vector2) -> f32 {
        let x = (self.x - other.x).abs();
        let y = (self.y - other.y).abs();
        (x * x + y * y).sqrt()
    }

    fn angle(&self, other: Vector2) -> f32 {
        let dot = self.dot(other);
        let mag = self.magnitude() * other.magnitude();
        (dot as f32 / mag).acos()
    }

    fn lerp(&self, other: Vector2, t: f32) -> Vector2 {
        Vector2 {
            x: (self.x as f32 + (other.x - self.x) * t),
            y: (self.y as f32 + (other.y - self.y) * t),
        }
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

    fn zero() -> Vector3 {
        Vector3 {
            x: 0.,
            y: 0.,
            z: 0.,
        }
    }

    fn one() -> Vector3 {
        Vector3 {
            x: 1.,
            y: 1.,
            z: 1.,
        }
    }

    fn cross(&self, other: Vector3) -> Vector3 {
        Vector3 {
            x: self.y * other.z - self.z * other.y,
            y: -(self.x * other.z - self.z * other.x),
            z: self.x * other.y - self.y * other.x,
        }
    }

    fn add(&self, other: Vector3) -> Vector3 {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    fn mul_scalar(&self, scalar: f32) -> Vector3 {
        Vector3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }

    fn sub(&self, other: Vector3) -> Vector3 {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    fn mul(&self, other: Vector3) -> Vector3 {
        Vector3 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }

    fn dot(&self, other: Vector3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn magnitude(&self) -> f32 {
        ((self.x * self.x + self.y * self.y + self.z * self.z) as f32).sqrt()
    }

    fn normalize(&self) -> Vector3 {
        let mag = self.magnitude();
        Vector3 {
            x: (self.x as f32 / mag),
            y: (self.y as f32 / mag),
            z: (self.z as f32 / mag),
        }
    }

    fn distance(&self, other: Vector3) -> f32 {
        let x = (self.x as f32 - other.x as f32).abs();
        let y = (self.y as f32 - other.y as f32).abs();
        let z = (self.z as f32 - other.z as f32).abs();
        (x * x + y * y + z * z).sqrt()
    }

    fn angle(&self, other: Vector3) -> f32 {
        let dot = self.dot(other);
        let mag = self.magnitude() * other.magnitude();
        (dot as f32 / mag).acos()
    }

    fn lerp(&self, other: Vector3, t: f32) -> Vector3 {
        Vector3 {
            x: (self.x as f32 + (other.x as f32 - self.x as f32) * t),
            y: (self.y as f32 + (other.y as f32 - self.y as f32) * t),
            z: (self.z as f32 + (other.z as f32 - self.z as f32) * t),
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

    fn zero() -> Vector4 {
        Vector4 {
            x: 0.,
            y: 0.,
            z: 0.,
            w: 0.,
        }
    }

    fn cross(&self, other: Vector4) -> Vector4 {
        Vector4 {
            x: self.y * other.z - self.z * other.y,
            y: -(self.x * other.z - self.z * other.x),
            z: self.x * other.y - self.y * other.x,
            w: 0.,
        }
    }

    fn one() -> Vector4 {
        Vector4 {
            x: 1.,
            y: 1.,
            z: 1.,
            w: 1.,
        }
    }

    fn add(&self, other: Vector4) -> Vector4 {
        Vector4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }

    fn sub(&self, other: Vector4) -> Vector4 {
        Vector4 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }

    fn mul(&self, other: Vector4) -> Vector4 {
        Vector4 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
            w: self.w * other.w,
        }
    }

    fn dot(&self, other: Vector4) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    fn normalize(&self) -> Vector4 {
        let mag = self.magnitude();
        Vector4 {
            x: (self.x as f32 / mag),
            y: (self.y as f32 / mag),
            z: (self.z as f32 / mag),
            w: (self.w as f32 / mag),
        }
    }

    fn distance(&self, other: Vector4) -> f32 {
        let x = (self.x as f32 - other.x as f32).abs();
        let y = (self.y as f32 - other.y as f32).abs();
        let z = (self.z as f32 - other.z as f32).abs();
        let w = (self.w as f32 - other.w as f32).abs();
        (x * x + y * y + z * z + w * w).sqrt()
    }

    fn angle(&self, other: Vector4) -> f32 {
        let dot = self.dot(other);
        let mag = self.magnitude() * other.magnitude();
        (dot as f32 / mag).acos()
    }

    fn lerp(&self, other: Vector4, t: f32) -> Vector4 {
        Vector4 {
            x: (self.x as f32 + (other.x as f32 - self.x as f32) * t),
            y: (self.y as f32 + (other.y as f32 - self.y as f32) * t),
            z: (self.z as f32 + (other.z as f32 - self.z as f32) * t),
            w: (self.w as f32 + (other.w as f32 - self.w as f32) * t),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Matrix44 {
    m: [[f32; 4]; 4],
}

impl Matrix44 {
    fn new(m: [[f32; 4]; 4]) -> Matrix44 {
        Matrix44 { m }
    }

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

    fn dotVec3(&self, other: Vector3) -> Vector3 {
        Vector3 {
            x: self.m[0][0] * other.x + self.m[0][1] * other.y + self.m[0][2] * other.z,
            y: self.m[1][0] * other.x + self.m[1][1] * other.y + self.m[1][2] * other.z,
            z: self.m[2][0] * other.x + self.m[2][1] * other.y + self.m[2][2] * other.z,
        }
    }

    fn dotVec4(&self, other: Vector4) -> Vector4 {
        Vector4 {
            x: self.m[0][0] * other.x
                + self.m[0][1] * other.y
                + self.m[0][2] * other.z
                + self.m[0][3] * other.w,
            y: self.m[1][0] * other.x
                + self.m[1][1] * other.y
                + self.m[1][2] * other.z
                + self.m[1][3] * other.w,
            z: self.m[2][0] * other.x
                + self.m[2][1] * other.y
                + self.m[2][2] * other.z
                + self.m[2][3] * other.w,
            w: self.m[3][0] * other.x
                + self.m[3][1] * other.y
                + self.m[3][2] * other.z
                + self.m[3][3] * other.w,
        }
    }

    fn zero() -> Matrix44 {
        Matrix44 {
            m: [[0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4]],
        }
    }

    fn add(&self, other: Matrix44) -> Matrix44 {
        let mut m = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = self.m[i][j] + other.m[i][j];
            }
        }
        Matrix44 { m }
    }

    fn sub(&self, other: Matrix44) -> Matrix44 {
        let mut m = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = self.m[i][j] - other.m[i][j];
            }
        }
        Matrix44 { m }
    }

    fn mul(&self, other: Matrix44) -> Matrix44 {
        let mut m = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = self.m[i][0] * other.m[0][j]
                    + self.m[i][1] * other.m[1][j]
                    + self.m[i][2] * other.m[2][j]
                    + self.m[i][3] * other.m[3][j];
            }
        }
        Matrix44 { m }
    }

    fn mul_vec3(&self, other: Vector3) -> Vector3 {
        let mut m = [0.0; 4];
        for i in 0..4 {
            m[i] = self.m[i][0] * other.x
                + self.m[i][1] * other.y
                + self.m[i][2] * other.z
                + self.m[i][3];
        }
        Vector3 {
            x: m[0],
            y: m[1],
            z: m[2],
        }
    }

    fn transpose(&self) -> Matrix44 {
        let mut m = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = self.m[j][i];
            }
        }
        Matrix44 { m }
    }

    fn dot(&self, other: Matrix44) -> f32 {
        let mut m = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = self.m[i][0] * other.m[0][j]
                    + self.m[i][1] * other.m[1][j]
                    + self.m[i][2] * other.m[2][j]
                    + self.m[i][3] * other.m[3][j];
            }
        }
        m[0][0] + m[1][1] + m[2][2] + m[3][3]
    }

    fn negate(&self) -> Matrix44 {
        let mut m = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = -self.m[i][j];
            }
        }
        Matrix44 { m }
    }

    //multiply a matrix by a vector

    fn mul_vec(&self, other: Vector4) -> Vector4 {
        let mut v = Vector4::new(0., 0., 0., 0.);
        v.x = (self.m[0][0] * other.x as f32
            + self.m[0][1] * other.y as f32
            + self.m[0][2] * other.z as f32
            + self.m[0][3] * other.w as f32);
        v.y = (self.m[1][0] * other.x as f32
            + self.m[1][1] * other.y as f32
            + self.m[1][2] * other.z as f32
            + self.m[1][3] * other.w as f32);
        v.z = (self.m[2][0] * other.x as f32
            + self.m[2][1] * other.y as f32
            + self.m[2][2] * other.z as f32
            + self.m[2][3] * other.w as f32);
        v.w = (self.m[3][0] * other.x as f32
            + self.m[3][1] * other.y as f32
            + self.m[3][2] * other.z as f32
            + self.m[3][3] * other.w as f32);
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
    projMatrix: Matrix44,
    viewMatrix: Matrix44,
}

impl Camera {
    fn new(
        startPos: Vector3,
        startRotation: Vector3,
        fov: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Camera {
        let mut camera = Camera {
            position: startPos,
            rotation: Vector3 {
                x: startRotation.x.to_radians(),
                y: startRotation.y.to_radians(),
                z: startRotation.z.to_radians(),
            },
            projMatrix: Matrix44::identity(),
            viewMatrix: Matrix44::identity(),
        };
        camera.createProjectionMatrix(fov, aspect, near, far);
        camera.calculateViewMatrix();
        camera
    }

    fn getPosition(&self) -> Vector3 {
        self.position
    }

    fn getRotation(&self) -> Vector3 {
        self.rotation
    }

    
    fn getForwardVector(&self) -> Vector3 {
        let forward = Vector3 {
            x: self.rotation.y.sin() * self.rotation.x.cos(),
            y: self.rotation.x.sin(),
            z: -self.rotation.y.cos() * self.rotation.x.cos(),
        }.normalize();
        
        forward
    }

    fn moveForward(&mut self, distance: f32) {
        let forward = self.getForwardVector();
        self.position.x += forward.x * distance;
        self.position.y += forward.y * distance;
        self.position.z += forward.z * distance;
        self.calculateViewMatrix();
    }

    fn moveRight(&mut self, distance: f32) {
        let forward = self.getForwardVector();
        let right = Vector3::new(-forward.z, 0., forward.x).normalize();
        self.position.x -= right.x * distance;
        self.position.y -= right.y * distance;
        self.position.z -= right.z * distance;
        self.calculateViewMatrix();
    }

    fn moveUp(&mut self, distance: f32) {
        self.position.y += distance;
        self.calculateViewMatrix();
    }


    fn rotateUp(&mut self, angle: f32) {
        self.rotation.x += angle;
        self.calculateViewMatrix();
    }

    fn rotateRight(&mut self, angle: f32) {
        self.rotation.y -= angle;
        self.calculateViewMatrix();
    }

    fn calculateViewMatrix(&mut self) {
        self.viewMatrix = Matrix44::identity();
        self.viewMatrix
            .rotate(Vector3::new(1., 0., 0.), self.rotation.x);
        self.viewMatrix
            .rotate(Vector3::new(0., 1., 0.), self.rotation.y);
        self.viewMatrix
            .rotate(Vector3::new(0., 0., 1.), self.rotation.z);
        self.viewMatrix
            .translate(-self.position.x, -self.position.y, -self.position.z);
    }

    fn getPvMatrix(&self) -> Matrix44 {
        self.projMatrix.mul(self.viewMatrix)
    }

    fn createProjectionMatrix(&mut self, fov: f32, aspect: f32, near: f32, far: f32) {
        let fov = fov.to_radians();
        let scale = 1.0 / (fov * 0.5).tan(); // Precompute to avoid duplicate calculation
        let fovY = scale * aspect;
        let fovX = scale;
        let f = far / (far - near);
        let nf = -(far * near) / (far - near);
        self.projMatrix = Matrix44 {
            m: [
                [fovX, 0.0, 0.0, 0.0],
                [0.0, fovY, 0.0, 0.0],
                [0.0, 0.0, f, nf],
                [0.0, 0.0, 1.0, 0.0],
            ],
        };
    }
}

fn transformVertex(vertex: Vector4, MvMatrix: Matrix44) -> Vector4 {
    let mut f: Vector4;

    f = MvMatrix.mul_vec(vertex);

    if f.w == 0. {
        f.w = 1e-5;
    }
    f.x = f.x / f.w;
    f.y = f.y / f.w;
    f.z = f.z / f.w;
    // f.w = 1.;

    f
}



fn main() {
    let aspectX = 16;
    let aspectY = 9;
    let rate = 20;
    let device_state = DeviceState::new();

    let start_time = Instant::now();

    // let aspectX = 64;
    // let aspectY = 48;
    // let rate = 1;

    //load model
    let input = BufReader::new(File::open("./monke.obj").unwrap());
    let mut model: Obj = load_obj(input).unwrap();

    let mut CubeFaces: Vec<Face> = Vec::new();

    // return;

    for i in 0..model.indices.len() / 3 {
        let mut face = Face {
            /*
                pub struct Vertex {
                /// Position vector of a vertex.
                pub position: [f32; 3],
                /// Normal vertor of a vertex.
                pub normal: [f32; 3],
            } */
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
            color: Vector3::new(1., 1., 1.),
        };

        CubeFaces.push(face);
    }

    //sort faces by distance to camera far to near
    CubeFaces.sort_by(|a, b| {
        let a = a.vertices[0].z + a.vertices[1].z + a.vertices[2].z;
        let b = b.vertices[0].z + b.vertices[1].z + b.vertices[2].z;
        b.partial_cmp(&a).unwrap()
    });

    let mut Camera = Camera::new(
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
        75.,
        (aspectX * rate) as f32 / 2. / (aspectY * rate) as f32,
        0.1,
        400.,
    );

    let mut angle = 0.;

    let mut transformation = Matrix44::new([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ]);

    let mut fb = FrameBuffer::new(aspectX * rate, aspectY * rate, Color { r: 0, g: 0, b: 0 });
    let fps = 64;

    //draw triangle

    let mut dt = 0;

    let v1 = Vector4 {
        x: 0.,
        y: -1.,
        z: 0.,
        w: 1.,
    };

    let v2 = Vector4 {
        x: -0.5,
        y: 1.,
        z: 0.,
        w: 1.,
    };

    let v3 = Vector4 {
        x: 0.5,
        y: 1.,
        z: 0.,
        w: 1.,
    };

    loop {
        let keys: Vec<Keycode> = device_state.get_keys();
        for key in keys.iter() {
            match key {
                Keycode::W => {
                    Camera.moveForward(0.1);
                }
                Keycode::S => {
                    Camera.moveForward(-0.1);
                }
                Keycode::A => {
                    Camera.moveRight(-0.1);
                }
                Keycode::D => {
                    Camera.moveRight(0.1);
                }
                Keycode::Q => {
                    Camera.moveUp(0.1);
                }
                Keycode::E => {
                    Camera.moveUp(-0.1);
                }
                Keycode::I => {
                    Camera.rotateUp(0.01);
                }
                Keycode::K => {
                    Camera.rotateUp(-0.01);
                }
                Keycode::J => {
                    Camera.rotateRight(-0.01);
                }
                Keycode::L => {
                    Camera.rotateRight(0.01);
                }
                _ => {}
            }
        }

        Camera.calculateViewMatrix();
        let PvMatrix = Camera.getPvMatrix();

        transformation = Matrix44::identity();

        transformation.translate(0.0, 0.0, -5.);

        transformation.rotate(
            Vector3 {
                x: 1.,
                y: 0.,
                z: 0.,
            },
            (180. as f32 ).to_radians(),
        );

        transformation.rotate(
            Vector3 {
                x: 0.,
                y: 1.,
                z: 0.,
            },
            angle,
        );

        // angle += 0.01;

        // let finalMatrix = PvMatrix.mul(transformation);

        //apply transformation
        // let fv1 = transformVertex(v1, finalMatrix);
        // let fv2 = transformVertex(v2, finalMatrix);
        // let fv3 = transformVertex(v3, finalMatrix);

        fb.clear();
        // fill_triangle(
        // Vector2 { x: fv1.x, y: fv1.y },
        // Vector2 { x: fv2.x, y: fv2.y },
        // Vector2 { x: fv3.x, y: fv3.y },
        // &mut fb,
        // );
        println!("ForwardVector: {:?}", Camera.getForwardVector());
        drawFaces(&CubeFaces, transformation, &mut fb, &Camera);
        fb.draw_frame();

        // println!("Model: {}", model.vertices.len());

        std::thread::sleep(std::time::Duration::from_millis(1000 / fps));
    }
}
