//import time
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};
use std::{thread, time};
use device_query::{DeviceQuery, DeviceState, Keycode};

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

    fn add(&self, other: Vector3) -> Vector3 {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
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

    fn calculateViewMatrix(&mut self) {
        self.viewMatrix = Matrix44::identity();
        // self.viewMatrix.rotateX(self.rotation.x);
        // self.viewMatrix.rotateY(self.rotation.y);
        // self.viewMatrix.rotateZ(self.rotation.z);
        self.viewMatrix
            .rotate(Vector3::new(1., 0., 0.), self.rotation.x);
        self.viewMatrix
            .rotate(Vector3::new(0., 1., 0.), self.rotation.y);
        self.viewMatrix
            .rotate(Vector3::new(0., 0., 1.), self.rotation.z);
        self.viewMatrix
            .translate(-self.position.x, -self.position.y, -self.position.z);
        // .rotateX(self.rotation.x)
        // .rotateY(self.rotation.y)
        // .rotateZ(self.rotation.z)
        // .translate(self.position.x, self.position.y, self.position.z);
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

struct FrameBuffer {
    front_buffer: Vec<u32>,
    back_buffer: Vec<u32>,
    width: usize,
    height: usize,
}

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
    //draws a line from v1 to v2
    //uses Bresenham's line algorithm
    let mut x1 = v1.x as i32;
    let mut y1 = v1.y as i32;
    let x2 = v2.x as i32;
    let y2 = v2.y as i32;

    let dx = (x2 - x1).abs();
    let dy = -(y2 - y1).abs();

    let mut sx = 1;
    let mut sy = 1;

    if x1 > x2 {
        sx = -1;
    }
    if y1 > y2 {
        sy = -1;
    }

    let mut err = dx + dy;
    let mut e2;

    loop {
        fb.set_pixel(
            x1 as usize,
            y1 as usize,
            Color {
                r: 255,
                g: 255,
                b: 255,
            },
        );

        if x1 == x2 && y1 == y2 {
            break;
        }
        e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x1 += sx;
        }
        if e2 <= dx {
            err += dx;
            y1 += sy;
        }
    }
}

fn fill_bottom_flat_triangle(
    v1: UsizeVector2,
    v2: UsizeVector2,
    v3: UsizeVector2,
    fb: &mut FrameBuffer,
) {
    //draws a triangle from v1 to v2 to v3
    //uses Bresenham's line algorithm
    //sort vertices by y value
    let mut v1 = v1;
    let mut v2 = v2;
    let mut v3 = v3;
    if v1.y > v2.y {
        let temp = v1;
        v1 = v2;
        v2 = temp;
    }
    if v1.y > v3.y {
        let temp = v1;
        v1 = v3;
        v3 = temp;
    }
    if v2.y > v3.y {
        let temp = v2;
        v2 = v3;
        v3 = temp;
    }

    let invslope1 = (v2.x as f32 - v1.x as f32) / (v2.y as f32 - v1.y as f32);
    let invslope2 = (v3.x as f32 - v1.x as f32) / (v3.y as f32 - v1.y as f32);

    let mut curx1 = v1.x as f32;
    let mut curx2 = v1.x as f32;

    for scanline_y in v1.y..=v2.y {
        draw_line(
            UsizeVector2 {
                x: curx1 as usize,
                y: scanline_y,
            },
            UsizeVector2 {
                x: curx2 as usize,
                y: scanline_y,
            },
            fb,
        );
        curx1 += invslope1;
        curx2 += invslope2;
    }
}

fn fill_top_flat_triangle(
    v1: UsizeVector2,
    v2: UsizeVector2,
    v3: UsizeVector2,
    fb: &mut FrameBuffer,
) {
    //draws a triangle from v1 to v2 to v3
    //uses Bresenham's line algorithm
    //sort vertices by y value
    let mut v1 = v1;
    let mut v2 = v2;
    let mut v3 = v3;
    if v1.y > v2.y {
        let temp = v1;
        v1 = v2;
        v2 = temp;
    }
    if v1.y > v3.y {
        let temp = v1;
        v1 = v3;
        v3 = temp;
    }
    if v2.y > v3.y {
        let temp = v2;
        v2 = v3;
        v3 = temp;
    }

    let invslope1 = (v3.x as f32 - v1.x as f32) / (v3.y as f32 - v1.y as f32);
    let invslope2 = (v3.x as f32 - v2.x as f32) / (v3.y as f32 - v2.y as f32);

    let mut curx1 = v3.x as f32;
    let mut curx2 = v3.x as f32;

    for scanline_y in (v1.y..=v3.y).rev() {
        draw_line(
            UsizeVector2 {
                x: curx1 as usize,
                y: scanline_y,
            },
            UsizeVector2 {
                x: curx2 as usize,
                y: scanline_y,
            },
            fb,
        );
        curx1 -= invslope1;
        curx2 -= invslope2;
    }
}

fn fill_triangle(vv1: Vector2, vv2: Vector2, vv3: Vector2, fb: &mut FrameBuffer) {
    let hwid = fb.width as f32 / 2.;
    let hhei = fb.height as f32 / 2.;

    let mut v1 = UsizeVector2 {
        x: ((vv1.x + 1.) * hwid) as usize,
        // y: ((vv1.y + 1.) * hhei) as usize,
        y: ((-vv1.y + 1.) * hhei) as usize,
    };

    let mut v2 = UsizeVector2 {
        x: ((vv2.x + 1.) * hwid) as usize,
        // y: ((vv2.y + 1.) * hhei) as usize,
        y: ((-vv2.y + 1.) * hhei) as usize,
    };

    let mut v3 = UsizeVector2 {
        x: ((vv3.x + 1.) * hwid) as usize,
        // y: ((vv3.y + 1.) * hhei) as usize,
        y: ((-vv3.y + 1.) * hhei) as usize,
    };

    if v1.y > v2.y {
        let temp = v1;
        v1 = v2;
        v2 = temp;
    }

    if v1.y > v3.y {
        let temp = v1;
        v1 = v3;
        v3 = temp;
    }

    if v2.y > v3.y {
        let temp = v2;
        v2 = v3;
        v3 = temp;
    }

    if v2.y == v3.y {
        fill_bottom_flat_triangle(v1, v2, v3, fb);
    } else if v1.y == v2.y {
        fill_top_flat_triangle(v1, v2, v3, fb);
    } else {
        let v4 = UsizeVector2 {
            x: (v1.x + ((v2.y - v1.y) / (v3.y - v1.y)) * (v3.x - v1.x)) as usize,
            y: v2.y as usize,
        };
        fill_bottom_flat_triangle(v1, v2, v4, fb);
        fill_top_flat_triangle(v2, v4, v3, fb);
    }
}

fn transformVertex(vertex: Vector4, MvMatrix: Matrix44) -> Vector4 {
    let mut f: Vector4;

    f = MvMatrix.mul_vec(vertex);

    f.x = f.x / f.w;
    f.y = f.y / f.w;
    f.z = f.z / f.w;
    f.w = 1.;

    f
}

fn main() {
    let aspectX = 16;
    let aspectY = 9;
    let rate = 10;

    let start_time = Instant::now();

    // let aspectX = 64;
    // let aspectY = 48;
    // let rate = 1;

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
        90.,
        (aspectX * rate) as f32 / 2. / (aspectY * rate) as f32,
        1e-3,
        1000.,
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
        Camera.calculateViewMatrix();
        let PvMatrix = Camera.getPvMatrix();

        transformation = Matrix44::identity();

        // transformation.translate(0.0, 0., 1.0);

        // transformation.translate(0.0, 0., -0.01);

        transformation.translate(0.0, 0.0, 1.5);

        transformation.rotate(
            Vector3 {
                x: 0.,
                y: 1.,
                z: 0.,
            },
            angle,
        );

        angle += 0.01;

        let finalMatrix = PvMatrix.mul(transformation);

        //apply transformation
        let fv1 = transformVertex(v1, finalMatrix);
        let fv2 = transformVertex(v2, finalMatrix);
        let fv3 = transformVertex(v3, finalMatrix);

        fb.clear();
        fill_triangle(
            Vector2 { x: fv1.x, y: fv1.y },
            Vector2 { x: fv2.x, y: fv2.y },
            Vector2 { x: fv3.x, y: fv3.y },
            &mut fb,
        );
        fb.draw_frame();
        // dt += 1 / 60;

        std::thread::sleep(std::time::Duration::from_millis(1000 / fps));
    }
}
