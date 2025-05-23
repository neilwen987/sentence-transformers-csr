import os
import sys

# Add project root directory to sys.path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    try:
        from sentence_transformers.sparse_encoder import SparseEncoder
        
        print("🔄 Loading SparseEncoder model...")
        
        # Load model directly through configuration folder
        model = SparseEncoder("sparse_encoder_model")
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model information:")
        print(f"   - Number of modules: {len(model._modules)}")
        print(f"   - Max sequence length: {model.max_seq_length}")
        print(f"   - Similarity function: {model.similarity_fn_name}")
        
        # Test encoding
        sentences = [
            "This is a test sentence.",
            "Hello world!",
            "Artificial intelligence is transforming the world."
        ]
        
        print(f"\n🧪 Testing encoding...")
        embeddings = model.encode(sentences)
        print(f"✅ Encoding successful!")
        print(f"   - Embedding shape: {embeddings.shape}")
        print(f"   - Embedding type: {type(embeddings)}")
        
        # Get sparsity statistics
        if hasattr(model, 'get_sparsity_stats'):
            stats = model.get_sparsity_stats(embeddings)
            print(f"\n📈 Sparsity statistics:")
            print(f"   - Number of rows: {stats['num_rows']}")
            print(f"   - Number of columns: {stats['num_cols']}")  
            print(f"   - Average non-zero elements: {stats['row_non_zero_mean']:.2f}")
            print(f"   - Average sparsity: {stats['row_sparsity_mean']:.4f}")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please check:")
        print("1. Whether configuration files are correctly generated")
        print("2. Whether sentence-transformers is properly installed")
        print("3. Whether there are environment compatibility issues")

if __name__ == "__main__":
    main() 