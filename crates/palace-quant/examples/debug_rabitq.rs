use palace_quant::rabitq::RaBitQIndex;
use rand::Rng;

fn main() {
    let dim = 128;
    let index = RaBitQIndex::new(dim, 42);

    // Create random vector
    let mut rng = rand::thread_rng();
    let v1: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let v2: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // Standard L2
    let sqr_dist = v1
        .iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>();

    println!("Standard Sqr L2: {}", sqr_dist);

    // RaBitQ encode
    let q = index.encode_query(&v1);
    let code = index.encode_multibit(&v2, 4);

    let (est_dist, _lb) = index.estimate_distance(&q, &code);

    println!("RaBitQ Estimate: {}", est_dist);
    println!(
        "RaBitQ Error: {:.2}%",
        (est_dist - sqr_dist).abs() / sqr_dist * 100.0
    );

    let q_res: Vec<f32> = v1.to_vec();
    let v_res: Vec<f32> = v2.to_vec();

    let q_norm = q_res.iter().map(|x| x * x).sum::<f32>().sqrt();
    let v_norm = v_res.iter().map(|x| x * x).sum::<f32>().sqrt();

    let ip = q_res
        .iter()
        .zip(v_res.iter())
        .map(|(a, b)| a * b)
        .sum::<f32>();
    let cos_sim = ip / (q_norm * v_norm);

    println!("Original Cosine Similarity: {}", cos_sim);
}
