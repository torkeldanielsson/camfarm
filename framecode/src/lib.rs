//! Frame identity code drawn into the top band of every test camera picture (camfarm) and read back (camcheck).
//!
//! The band is `ROWS` x `COLS` cells over the full picture width and the top tenth of the picture height.
//! Cell size follows the picture size (width / 40, height / 30), so the code survives any uniform scaling
//! (e.g. 1280x720 or 1920x1080 cameras composed at 960x540) and lossy coding (cells are large, black or white,
//! and read as the mean of their central half). 120 cells carry:
//!
//! | cells   | content                                               |
//! |---------|-------------------------------------------------------|
//! | 0..8    | sync pattern 10101100 (also the black/white reference) |
//! | 8..16   | camera id                                             |
//! | 16..48  | frame number                                          |
//! | 48..96  | capture time, microseconds since the Unix epoch, low 48 bits |
//! | 96..112 | CRC-16/CCITT-FALSE of the 13 payload bytes            |
//! | 112..120| sync pattern 01010011                                 |

pub const COLS: u32 = 40;
pub const ROWS: u32 = 3;
pub const BITS: usize = (COLS * ROWS) as usize;
pub const LUMA_BLACK: u8 = 16;
pub const LUMA_WHITE: u8 = 235;
const SYNC_HEAD: u8 = 0b1010_1100;
const SYNC_TAIL: u8 = 0b0101_0011;
const TIME_MASK: u64 = (1 << 48) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameId {
    pub camera: u8,
    pub frame: u32,
    /// Capture time in microseconds since the Unix epoch (only the low 48 bits are carried).
    pub capture_us: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub cell_w: u32,
    pub cell_h: u32,
}

impl Geometry {
    pub fn for_picture(width: u32, height: u32) -> Self {
        Self { cell_w: width / COLS, cell_h: height / 30 }
    }

    /// Rows of the picture (from the top) that hold the code band.
    pub fn band_rows(&self) -> u32 {
        self.cell_h * ROWS
    }
}

fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xffff;
    for &b in data {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 { (crc << 1) ^ 0x1021 } else { crc << 1 };
        }
    }
    crc
}

fn payload(id: &FrameId) -> [u8; 13] {
    let mut p = [0u8; 13];
    p[0] = id.camera;
    p[1..5].copy_from_slice(&id.frame.to_be_bytes());
    p[5..11].copy_from_slice(&(id.capture_us & TIME_MASK).to_be_bytes()[2..8]);
    p[11] = SYNC_HEAD;
    p[12] = SYNC_TAIL;
    p
}

/// The 120 code bits, most significant bit of each field first, packed into four u32 words (bit i of the code
/// is bit (31 - i % 32) of word i / 32).
pub fn encode(id: &FrameId) -> [u32; 4] {
    let p = payload(id);
    let crc = crc16(&p);
    let mut bytes = [0u8; 15];
    bytes[0] = SYNC_HEAD;
    bytes[1..12].copy_from_slice(&p[0..11]);
    bytes[12..14].copy_from_slice(&crc.to_be_bytes());
    bytes[14] = SYNC_TAIL;
    let mut words = [0u32; 4];
    for (i, b) in bytes.iter().enumerate() {
        words[i / 4] |= (*b as u32) << (24 - 8 * (i % 4));
    }
    words
}

pub fn bit(words: &[u32; 4], i: usize) -> bool {
    (words[i / 32] >> (31 - i % 32)) & 1 == 1
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecodeError {
    /// Black and white reference cells are too close: no code band in this picture.
    NoContrast { black: f32, white: f32 },
    BadSync,
    BadCrc,
}

/// Decode from the luma plane rows that hold the band (`luma[y * stride + x]`, y from the top of the picture).
pub fn decode(luma: &[u8], stride: usize, width: u32, height: u32) -> Result<FrameId, DecodeError> {
    let g = Geometry::for_picture(width, height);
    let mut means = [0f32; BITS];
    for (i, m) in means.iter_mut().enumerate() {
        let cx = (i as u32 % COLS) * g.cell_w;
        let cy = (i as u32 / COLS) * g.cell_h;
        let (x0, x1) = (cx + g.cell_w / 4, cx + g.cell_w * 3 / 4);
        let (y0, y1) = (cy + g.cell_h / 4, cy + g.cell_h * 3 / 4);
        let mut sum = 0u32;
        for y in y0..y1.max(y0 + 1) {
            let row = &luma[y as usize * stride..];
            for x in x0..x1.max(x0 + 1) {
                sum += row[x as usize] as u32;
            }
        }
        *m = sum as f32 / ((y1.max(y0 + 1) - y0) * (x1.max(x0 + 1) - x0)) as f32;
    }
    let sync: Vec<(usize, bool)> = (0..8)
        .map(|k| (k, (SYNC_HEAD >> (7 - k)) & 1 == 1))
        .chain((0..8).map(|k| (112 + k, (SYNC_TAIL >> (7 - k)) & 1 == 1)))
        .collect();
    let mean_of = |want: bool| {
        let v: Vec<f32> = sync.iter().filter(|(_, b)| *b == want).map(|(i, _)| means[*i]).collect();
        v.iter().sum::<f32>() / v.len() as f32
    };
    let (black, white) = (mean_of(false), mean_of(true));
    if white - black < 40.0 {
        return Err(DecodeError::NoContrast { black, white });
    }
    let threshold = (black + white) / 2.0;
    let mut bytes = [0u8; 15];
    for i in 0..BITS {
        if means[i] > threshold {
            bytes[i / 8] |= 1 << (7 - i % 8);
        }
    }
    if bytes[0] != SYNC_HEAD || bytes[14] != SYNC_TAIL {
        return Err(DecodeError::BadSync);
    }
    let mut p = [0u8; 13];
    p[0..11].copy_from_slice(&bytes[1..12]);
    p[11] = SYNC_HEAD;
    p[12] = SYNC_TAIL;
    if crc16(&p) != u16::from_be_bytes([bytes[12], bytes[13]]) {
        return Err(DecodeError::BadCrc);
    }
    let mut t = [0u8; 8];
    t[2..8].copy_from_slice(&p[5..11]);
    Ok(FrameId { camera: p[0], frame: u32::from_be_bytes([p[1], p[2], p[3], p[4]]), capture_us: u64::from_be_bytes(t) })
}

/// Full capture time from the carried low 48 bits, given any time within ~4 years of it (e.g. the arrival time).
pub fn unwrap_capture_us(carried: u64, near_us: u64) -> u64 {
    let base = near_us & !TIME_MASK;
    let candidates = [base.wrapping_sub(1 << 48) | carried, base | carried, (base + (1 << 48)) | carried];
    *candidates.iter().min_by_key(|c| c.abs_diff(near_us)).unwrap()
}

/// Render the band into a luma plane (CPU reference implementation, used by the tests and as the spec for the
/// GPU shader).
pub fn draw(luma: &mut [u8], stride: usize, width: u32, height: u32, id: &FrameId) {
    let g = Geometry::for_picture(width, height);
    let words = encode(id);
    for y in 0..g.band_rows() {
        for x in 0..width {
            let col = (x / g.cell_w).min(COLS - 1);
            let i = ((y / g.cell_h) * COLS + col) as usize;
            luma[y as usize * stride + x as usize] = if bit(&words, i) { LUMA_WHITE } else { LUMA_BLACK };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picture(w: u32, h: u32, id: &FrameId) -> Vec<u8> {
        let mut p = vec![128u8; (w * h) as usize];
        draw(&mut p, w as usize, w, h, id);
        p
    }

    fn scale(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
        // box filter, like a GPU minification without mipmaps
        let mut d = vec![0u8; (dw * dh) as usize];
        for y in 0..dh {
            for x in 0..dw {
                let (x0, x1) = (x * sw / dw, ((x + 1) * sw / dw).max(x * sw / dw + 1));
                let (y0, y1) = (y * sh / dh, ((y + 1) * sh / dh).max(y * sh / dh + 1));
                let mut s = 0u32;
                for yy in y0..y1 {
                    for xx in x0..x1 {
                        s += src[(yy * sw + xx) as usize] as u32;
                    }
                }
                d[(y * dw + x) as usize] = (s / ((x1 - x0) * (y1 - y0))) as u8;
            }
        }
        d
    }

    #[test]
    fn roundtrip_and_scaling() {
        let id = FrameId { camera: 37, frame: 0x01234567, capture_us: 1_791_550_000_123_456 };
        for (w, h) in [(1280, 720), (1920, 1080), (960, 540), (640, 360)] {
            let p = picture(w, h, &id);
            assert_eq!(decode(&p, w as usize, w, h).unwrap(), FrameId { capture_us: id.capture_us & TIME_MASK, ..id });
            let s = scale(&p, w, h, 960, 540);
            assert_eq!(decode(&s, 960, 960, 540).unwrap().frame, id.frame, "{w}x{h} scaled to 960x540");
        }
    }

    #[test]
    fn noise_and_contrast_loss() {
        let id = FrameId { camera: 3, frame: 99, capture_us: 42 };
        let mut p = picture(960, 540, &id);
        let mut state = 12345u32;
        for v in p.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = ((state >> 24) as i32 - 128) / 3; // +-42 noise
            *v = (*v as i32 / 2 + 64 + n).clamp(0, 255) as u8; // half the contrast, plus noise
        }
        assert_eq!(decode(&p, 960, 960, 540).unwrap().frame, 99);
    }

    #[test]
    fn rejects_garbage() {
        let p = vec![128u8; 960 * 540];
        assert!(matches!(decode(&p, 960, 960, 540), Err(DecodeError::NoContrast { .. })));
        let id = FrameId { camera: 1, frame: 7, capture_us: 9 };
        let mut p = picture(960, 540, &id);
        let g = Geometry::for_picture(960, 540);
        for y in 0..g.cell_h {
            for x in 20 * g.cell_w..21 * g.cell_w {
                p[(y * 960 + x) as usize] ^= 0xff; // flip one payload cell
            }
        }
        assert_eq!(decode(&p, 960, 960, 540), Err(DecodeError::BadCrc));
    }

    #[test]
    fn unwrap_time() {
        let full = 1_791_550_000_123_456u64;
        assert_eq!(unwrap_capture_us(full & TIME_MASK, full + 2_000_000), full);
        assert_eq!(unwrap_capture_us(full & TIME_MASK, full - 2_000_000), full);
    }
}
