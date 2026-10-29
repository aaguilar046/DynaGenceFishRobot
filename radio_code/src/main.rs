use rustfft::{num_complex::Complex, FftPlanner};

const N: usize = 64; // subcarriers
const L: usize = 16; // cyclic prefix

fn qpsk_map(bits: &[u8]) -> Vec<Complex<f32>> {
    let s = 1.0 / 2f32.sqrt();
    bits.chunks(2)
        .map(|b| Complex::new((1.0 - 2.0 * b[0] as f32) * s,
                              (1.0 - 2.0 * b[1] as f32) * s))
        .collect()
}

fn qpsk_demap(symbols: &[Complex<f32>]) -> Vec<u8> {
    symbols.iter()
        .flat_map(|s| [(s.re < 0.0) as u8, (s.im < 0.0) as u8])
        .collect()
}

fn ofdm_tx(symbols: &[Complex<f32>]) -> Vec<Complex<f32>> {
    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(N);

    let mut buf = symbols.to_vec();
    ifft.process(&mut buf);                    // in-place IFFT

    let scale = 1.0 / (N as f32).sqrt();       // rustfft doesn't normalize
    for x in buf.iter_mut() { *x *= scale; }

    let mut tx = buf[N - L..].to_vec();        // cyclic prefix
    tx.extend_from_slice(&buf);
    tx
}

fn ofdm_rx(rx: &[Complex<f32>]) -> Vec<Complex<f32>> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N);

    let mut buf = rx[L..L + N].to_vec();       // drop prefix
    fft.process(&mut buf);                     // in-place FFT

    let scale = 1.0 / (N as f32).sqrt();
    for x in buf.iter_mut() { *x *= scale; }
    buf
}

fn main() {
    let bits: Vec<u8> = (0..2 * N).map(|i| ((i * 7 + 3) % 5 % 2) as u8).collect();

    let tx = ofdm_tx(&qpsk_map(&bits));
    // no channel yet: rx = tx
    let bits_out = qpsk_demap(&ofdm_rx(&tx));

    assert_eq!(bits, bits_out);
    println!("round trip OK");
}