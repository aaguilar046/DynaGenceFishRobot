use num_complex::Complex32;

/// Maps a slice of bits to a vector of QPSK complex numbers.
/// Assumes input length is even.
fn bits_to_qpsk(bits: &[u8]) -> Vec<Complex32> {
    // A standard QPSK normalization factor: 1 / sqrt(2)
    let norm = 1.0 / (2.0_f32).sqrt(); 

    bits.chunks_exact(2)
        .map(|pair| {
            let b1 = pair[0];
            let b2 = pair[1];

            // Gray-coded mapping:
            // 00 -> (-1, -1), 01 -> (-1, +1), 11 -> (+1, +1), 10 -> (+1, -1)
            let i = if b1 == 1 { 1.0 } else { -1.0 };
            let q = if b2 == 1 { 1.0 } else { -1.0 };

            // Return the scaled complex number (I + jQ)
            Complex32::new(i * norm, q * norm)
        })
        .collect()
}

fn main() {
    // Example bitstream: [0, 0, 0, 1, 1, 1, 1, 0]
    let bits = vec![0, 0, 0, 1, 1, 1, 1, 0];
    let symbols = bits_to_qpsk(&bits);

    println!("Input bits: {:?}", bits);
    for (i, symbol) in symbols.iter().enumerate() {
        println!("Symbol {}: {:.3} + {:.3}j", i, symbol.re, symbol.im);
    }
}
