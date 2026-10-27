//! Adapted from `../quantorch/examples/gan_surface_generation.py`.
//!
//! The Python example trains a Generator/Discriminator pair with
//! `train_gan` (Adam loops over epochs). quantburn does not port training
//! loops — only the numeric forward-pass core of the GAN was converted (see
//! `src/gans.rs` and the "Not converted" note in README.md) — so this
//! example exercises that core instead: it builds the same architectures
//! (Generator 100 -> 128 -> 256 -> 10 with ReLU/Tanh; Discriminator
//! 10 -> 256 -> 128 -> 1 with LeakyReLU/Sigmoid), runs deterministic
//! forward passes on latent noise and "real" data drawn from the shared
//! deterministic sampler, and evaluates the BCE losses that the Python
//! training loop minimizes.

use quantburn::gans::{bce_loss, discriminator_forward, generator_forward, sigmoid, tanh};
use quantburn::rng::Xorshift64Star;

const INPUT_DIM: usize = 100;
const OUTPUT_DIM: usize = 10;
const BATCH_SIZE: usize = 32;

fn weight_matrix(rng: &mut Xorshift64Star, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|_| (0..cols).map(|_| 0.1 * rng.next_std_normal()).collect())
        .collect()
}

fn bias_vec(rng: &mut Xorshift64Star, n: usize) -> Vec<f64> {
    (0..n).map(|_| 0.1 * rng.next_std_normal()).collect()
}

fn main() {
    let mut rng = Xorshift64Star::new(42);

    // Deterministic stand-ins for the (untrained) torch default weights.
    let g_w1 = weight_matrix(&mut rng, 128, INPUT_DIM);
    let g_b1 = bias_vec(&mut rng, 128);
    let g_w2 = weight_matrix(&mut rng, 256, 128);
    let g_b2 = bias_vec(&mut rng, 256);
    let g_w3 = weight_matrix(&mut rng, OUTPUT_DIM, 256);
    let g_b3 = bias_vec(&mut rng, OUTPUT_DIM);

    let d_w1 = weight_matrix(&mut rng, 256, OUTPUT_DIM);
    let d_b1 = bias_vec(&mut rng, 256);
    let d_w2 = weight_matrix(&mut rng, 128, 256);
    let d_b2 = bias_vec(&mut rng, 128);
    let d_w3 = weight_matrix(&mut rng, 1, 128);
    let d_b3 = bias_vec(&mut rng, 1);

    // One batch of latent noise z ~ N(0, I) and one batch of "real" data,
    // both from the deterministic sampler.
    let latent = rng.normals(BATCH_SIZE * INPUT_DIM);
    let real: Vec<Vec<f64>> = (0..BATCH_SIZE)
        .map(|_| rng.normals(OUTPUT_DIM))
        .collect();

    // Generator forward: Linear -> ReLU -> Linear -> ReLU -> Linear -> Tanh.
    let fake: Vec<Vec<f64>> = (0..BATCH_SIZE)
        .map(|b| {
            let z = &latent[b * INPUT_DIM..(b + 1) * INPUT_DIM];
            tanh(&generator_forward(
                z, &g_w1, &g_b1, &g_w2, &g_b2, &g_w3, &g_b3,
            ))
        })
        .collect();

    // Discriminator forward: Linear -> LeakyReLU -> Linear -> LeakyReLU ->
    // Linear -> Sigmoid.
    // discriminator_forward returns a scalar logit; sigmoid it as a 1-element
    // slice (the final Linear -> Sigmoid stage of the Python Discriminator).
    let score =
        |x: &[f64]| sigmoid(&[discriminator_forward(x, &d_w1, &d_b1, &d_w2, &d_b2, &d_w3, &d_b3)])[0];

    let fake_preds: Vec<f64> = fake.iter().map(|x| score(x)).collect();
    let real_preds: Vec<f64> = real.iter().map(|x| score(x)).collect();

    let ones = vec![1.0; BATCH_SIZE];
    let zeros = vec![0.0; BATCH_SIZE];

    // BCE losses on the real and generated batches — the quantities
    // train_gan backpropagates in the Python original.
    let d_loss_real = bce_loss(&real_preds, &ones);
    let d_loss_fake = bce_loss(&fake_preds, &zeros);
    let g_loss = bce_loss(&fake_preds, &ones);

    println!("Sample generated surface row 0: {:?}", fake[0]);
    println!("Discriminator on real (mean): {:.6}", real_preds.iter().sum::<f64>() / BATCH_SIZE as f64);
    println!("Discriminator on fake (mean): {:.6}", fake_preds.iter().sum::<f64>() / BATCH_SIZE as f64);
    println!("D loss (real): {d_loss_real:.6}");
    println!("D loss (fake): {d_loss_fake:.6}");
    println!("G loss: {g_loss:.6}");
}
