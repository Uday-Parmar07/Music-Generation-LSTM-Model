from src.components.generate import MelodyGenerator


if __name__ == "__main__":
    generator = MelodyGenerator()
    seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"
    melody = generator.generate_melody(seed=seed, num_steps=500, temperature=0.3)
    generator.save_melody(melody)
    print("Generated melody saved to main/mel.mid")
