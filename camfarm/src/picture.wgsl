// One test camera picture, written as NV12 (Y plane, then interleaved UV) in packed u32 words.
//
// Top tenth: the frame code band (see the framecode crate). Below: an animated value-noise field with domain
// warping and a few drifting discs (motion for the encoder's inter prediction), plus per-pixel white noise
// (entropy: more noise needs more bits). The smooth field is rendered at a quarter of the picture size by
// `field` and sampled bilinearly by `main`.
//
// Integer hash: "lowbias32" by Chris Wellons (https://nullprogram.com/blog/2018/07/31/), public domain.

struct Params {
    width: u32,
    height: u32,
    band_rows: u32,
    cell_w: u32,
    cell_h: u32,
    cols: u32,
    frame: u32,
    seed: u32,
    time: f32,
    noise: f32,
    hue: f32,
    speed: f32,
    bits: vec4<u32>,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> out_words: array<u32>;
@group(0) @binding(2) var<storage, read_write> low: array<u32>;

fn lowbias32(x0: u32) -> u32 {
    var x = x0;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn unit(h: u32) -> f32 {
    return f32(h >> 8u) / 16777216.0;
}

// Value noise on an integer lattice, quintic interpolation, range 0..1.
fn vnoise(q: vec2<f32>, salt: u32) -> f32 {
    let i = floor(q);
    let f = q - i;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let ix = u32(i32(i.x) + 65536);
    let iy = u32(i32(i.y) + 65536);
    let a = unit(lowbias32(ix * 73856093u ^ iy * 19349663u ^ salt));
    let b = unit(lowbias32((ix + 1u) * 73856093u ^ iy * 19349663u ^ salt));
    let c = unit(lowbias32(ix * 73856093u ^ (iy + 1u) * 19349663u ^ salt));
    let d = unit(lowbias32((ix + 1u) * 73856093u ^ (iy + 1u) * 19349663u ^ salt));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(q: vec2<f32>, salt: u32) -> f32 {
    return 0.55 * vnoise(q, salt) + 0.3 * vnoise(q * 2.03 + 17.0, salt + 1u) + 0.15 * vnoise(q * 4.11 + 41.0, salt + 2u);
}

fn hue_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = fract(vec3<f32>(h, h + 2.0 / 3.0, h + 1.0 / 3.0)) * 6.0;
    let c = clamp(abs(k - 3.0) - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
    return v * mix(vec3<f32>(1.0), c, s);
}

fn field_rgb(uv: vec2<f32>) -> vec3<f32> {
    let t = p.time * p.speed;
    let aspect = f32(p.width) / f32(p.height);
    let q = vec2<f32>(uv.x * aspect, uv.y) * 3.0;
    let warp = vec2<f32>(fbm(q + vec2<f32>(t * 0.7, 0.0), p.seed), fbm(q + vec2<f32>(0.0, t * 0.5) + 9.0, p.seed + 7u));
    let n = fbm(q + 2.5 * warp + vec2<f32>(t * 0.3, -t * 0.2), p.seed + 13u);
    var c = hue_rgb(p.hue + 0.35 * n, 0.75, 0.25 + 0.6 * n);
    // three drifting discs
    for (var k = 0u; k < 3u; k = k + 1u) {
        let fk = f32(k);
        let centre = vec2<f32>(0.5 + 0.35 * sin(t * (0.9 + 0.3 * fk) + fk * 2.1), 0.55 + 0.3 * cos(t * (0.7 + 0.2 * fk) + fk * 1.3));
        let d = length((uv - centre) * vec2<f32>(aspect, 1.0));
        let edge = smoothstep(0.11, 0.1, d);
        c = mix(c, hue_rgb(p.hue + 0.5 + 0.15 * fk, 0.6, 0.9), edge);
    }
    return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(64, 1, 1)
fn field(@builtin(global_invocation_id) g: vec3<u32>) {
    let lw = p.width / 4u;
    let lh = p.height / 4u;
    if (g.x >= lw || g.y >= lh) {
        return;
    }
    let c = field_rgb(vec2<f32>((f32(g.x) + 0.5) / f32(lw), (f32(g.y) + 0.5) / f32(lh)));
    low[g.y * lw + g.x] = u32(c.r * 255.0) | (u32(c.g * 255.0) << 8u) | (u32(c.b * 255.0) << 16u);
}

fn low_texel(x: u32, y: u32) -> vec3<f32> {
    let lw = p.width / 4u;
    let v = low[min(y, p.height / 4u - 1u) * lw + min(x, lw - 1u)];
    return vec3<f32>(f32(v & 0xffu), f32((v >> 8u) & 0xffu), f32((v >> 16u) & 0xffu)) / 255.0;
}

fn low_rgb(u: f32, v: f32) -> vec3<f32> {
    let fx = max(u * f32(p.width / 4u) - 0.5, 0.0);
    let fy = max(v * f32(p.height / 4u) - 0.5, 0.0);
    let x0 = u32(fx);
    let y0 = u32(fy);
    let ax = fx - f32(x0);
    let ay = fy - f32(y0);
    let top = mix(low_texel(x0, y0), low_texel(x0 + 1u, y0), ax);
    let bot = mix(low_texel(x0, y0 + 1u), low_texel(x0 + 1u, y0 + 1u), ax);
    return mix(top, bot, ay);
}

fn white_noise(x: u32, y: u32, salt: u32) -> f32 {
    return unit(lowbias32(x * 374761393u ^ y * 668265263u ^ p.frame * 2246822519u ^ p.seed * 3266489917u ^ salt)) * 2.0 - 1.0;
}

fn code_bit(i: u32) -> bool {
    return ((p.bits[i / 32u] >> (31u - i % 32u)) & 1u) == 1u;
}

fn luma(x: u32, y: u32) -> u32 {
    if (y < p.band_rows) {
        let col = min(x / p.cell_w, p.cols - 1u);
        if (code_bit((y / p.cell_h) * p.cols + col)) {
            return 235u;
        }
        return 16u;
    }
    let c = low_rgb((f32(x) + 0.5) / f32(p.width), (f32(y) + 0.5) / f32(p.height));
    let v = 16.0 + 219.0 * dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)) + p.noise * white_noise(x, y, 0u);
    return u32(clamp(v, 16.0, 235.0));
}

fn chroma(cx: u32, cy: u32) -> vec2<u32> {
    if (cy * 2u < p.band_rows) {
        return vec2<u32>(128u, 128u);
    }
    let c = low_rgb((f32(cx) * 2.0 + 1.0) / f32(p.width), (f32(cy) * 2.0 + 1.0) / f32(p.height));
    let yl = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    let u = 128.0 + 224.0 * (c.b - yl) / 1.8556 + 0.5 * p.noise * white_noise(cx, cy, 1u);
    let v = 128.0 + 224.0 * (c.r - yl) / 1.5748 + 0.5 * p.noise * white_noise(cx, cy, 2u);
    return vec2<u32>(u32(clamp(u, 16.0, 240.0)), u32(clamp(v, 16.0, 240.0)));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let words_per_row = p.width / 4u;
    if (g.x >= words_per_row) {
        return;
    }
    let row = g.y;
    if (row < p.height) {
        let x = g.x * 4u;
        out_words[row * words_per_row + g.x] =
            luma(x, row) | (luma(x + 1u, row) << 8u) | (luma(x + 2u, row) << 16u) | (luma(x + 3u, row) << 24u);
    } else {
        let cy = row - p.height;
        if (cy >= p.height / 2u) {
            return;
        }
        let a = chroma(g.x * 2u, cy);
        let b = chroma(g.x * 2u + 1u, cy);
        out_words[(p.height + cy) * words_per_row + g.x] = a.x | (a.y << 8u) | (b.x << 16u) | (b.y << 24u);
    }
}
