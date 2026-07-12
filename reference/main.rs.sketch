use parakeet_rs::{Parakeet, Transcriber, TimestampMode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize the model from your local directory
    let mut parakeet = Parakeet::from_pretrained(".")?;

    // 2. Run the transcription with Word-level timestamp mode enabled
    let result = parakeet.transcribe_samples(
        audio,        // Your Vec<f32> or &[f32] audio buffer
        sample_rate,  // e.g., 16000
        channels,     // e.g., 1
        Some(TimestampMode::Words)
    )?;

    // 3. Print the full aggregated transcription
    println!("Transcription: {}", result.text);
    println!("--- Word Timestamps ---");

    // 4. Iterate through the individual word segments
    // (Note: Struct fields may vary slightly depending on your exact crate version,
    // but typically follow the segment/word timestamp pattern)
    if let Some(words) = result.words {
        for word_info in words {
            println!(
                "Word: [{:<12}] | Start: {:.2}s | End: {:.2}s",
                word_info.word, 
                word_info.start, 
                word_info.end
            );
        }
    } 
    else {
        println!("No word-level timestamps returned.");
    }

    Ok(())
}
