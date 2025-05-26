#!/usr/bin/env python3
"""
vLLM Troubleshooting Script
Diagnoses common issues and provides fixes for vLLM initialization problems
"""

import os
import sys
import subprocess
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_system_requirements():
    """Check system requirements and compatibility"""
    print("🔍 Checking System Requirements...")
    
    # Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major != 3 or python_version.minor < 8 or python_version.minor > 11:
        print("⚠️  WARNING: Python version should be 3.8-3.11 for best compatibility")
    
    # PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("❌ No CUDA GPUs detected")
    
    return cuda_available


def fix_pytorch_dynamo():
    """Fix PyTorch dynamo compilation issues"""
    print("\n🔧 Applying PyTorch Dynamo Fixes...")
    
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.verbose = False
        print("✅ PyTorch dynamo error suppression enabled")
    except ImportError:
        print("⚠️  PyTorch dynamo not available in this version")
    
    # Set environment variables
    env_fixes = {
        "TORCH_LOGS": "",  # Clear verbose logging
        "TORCHDYNAMO_VERBOSE": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "CUDA_VISIBLE_DEVICES": "0",  # Start with single GPU
        "TOKENIZERS_PARALLELISM": "false",  # Avoid tokenizer warnings
    }
    
    for key, value in env_fixes.items():
        os.environ[key] = value
        print(f"Set {key}={value}")


def test_vllm_import():
    """Test vLLM import and basic functionality"""
    print("\n📦 Testing vLLM Import...")
    
    try:
        import vllm
        print(f"✅ vLLM version: {vllm.__version__}")
        
        from vllm import LLM, SamplingParams
        print("✅ vLLM classes imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ vLLM import failed: {e}")
        print("💡 Try: pip install vllm")
        return False
    except Exception as e:
        print(f"❌ vLLM import error: {e}")
        return False


def test_minimal_vllm():
    """Test minimal vLLM configuration"""
    print("\n🧪 Testing Minimal vLLM Configuration...")
    
    try:
        from vllm import LLM, SamplingParams
        
        # Minimal configuration for testing
        print("Attempting minimal vLLM initialization...")
        llm = LLM(
            model="facebook/opt-125m",  # Very small model for testing
            tensor_parallel_size=1,
            gpu_memory_utilization=0.3,
            max_model_len=512,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            trust_remote_code=True,
            disable_log_stats=True,
        )
        
        print("✅ Minimal vLLM engine created successfully")
        
        # Test generation
        sampling_params = SamplingParams(temperature=0, max_tokens=10)
        outputs = llm.generate(["Hello world"], sampling_params)
        print(f"✅ Test generation successful: {outputs[0].outputs[0].text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal vLLM test failed: {e}")
        return False


def provide_fixes():
    """Provide specific fixes for common issues"""
    print("\n🛠️  Common Fixes:")
    
    print("""
1. **PyTorch Dynamo Issues**: Add this to your script:
   ```python
   import torch._dynamo
   torch._dynamo.config.suppress_errors = True
   ```

2. **CUDA Memory Issues**: Reduce GPU memory utilization:
   ```python
   gpu_memory_utilization=0.6  # Instead of 0.85
   ```

3. **Multi-GPU Issues**: Start with single GPU:
   ```python
   tensor_parallel_size=1
   os.environ["CUDA_VISIBLE_DEVICES"] = "0"
   ```

4. **Compilation Issues**: Force eager execution:
   ```python
   enforce_eager=True
   disable_custom_all_reduce=True
   ```

5. **Model Loading Issues**: Try smaller model first:
   ```python
   model="microsoft/DialoGPT-small"  # Test with smaller model
   ```
    """)


def install_compatible_versions():
    """Install compatible versions of key packages"""
    print("\n📥 Installing Compatible Versions...")
    
    commands = [
        "pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121",
        "pip install transformers==4.35.0",
        "pip install vllm==0.2.7",
        "pip install accelerate==0.24.0",
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✅ Success")
            else:
                print(f"❌ Failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("⏰ Command timed out")
        except Exception as e:
            print(f"❌ Error: {e}")


def create_test_script():
    """Create a test script with all fixes applied"""
    test_script = """
import os
import torch

# Apply fixes before importing vLLM
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Set environment variables
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from vllm import LLM, SamplingParams
    
    # Conservative configuration
    print("🚀 Initializing vLLM with conservative settings...")
    llm = LLM(
        model="defog/sqlcoder-7b-2",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        max_model_len=2048,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    
    print("✅ vLLM initialized successfully!")
    
    # Test generation
    sampling_params = SamplingParams(temperature=0, max_tokens=50)
    test_prompt = "SELECT * FROM"
    outputs = llm.generate([test_prompt], sampling_params)
    
    print(f"🎯 Test output: {outputs[0].outputs[0].text}")
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please check the troubleshooting guide above.")
"""
    
    with open("test_vllm_fixed.py", "w") as f:
        f.write(test_script)
    
    print("💾 Created test_vllm_fixed.py")
    print("Run with: python test_vllm_fixed.py")


def main():
    """Main troubleshooting function"""
    print("🔧 vLLM Troubleshooting Script")
    print("=" * 50)
    
    # Step 1: Check system requirements
    cuda_available = check_system_requirements()
    
    # Step 2: Apply fixes
    fix_pytorch_dynamo()
    
    # Step 3: Test vLLM import
    if not test_vllm_import():
        print("\n❌ vLLM import failed. Please install vLLM first.")
        return
    
    # Step 4: Test minimal configuration
    if test_minimal_vllm():
        print("\n✅ vLLM is working! The issue might be with your specific configuration.")
    else:
        print("\n❌ vLLM minimal test failed. Trying fixes...")
    
    # Step 5: Provide fixes
    provide_fixes()
    
    # Step 6: Create test script
    create_test_script()
    
    print("\n🎯 Next Steps:")
    print("1. Run: python test_vllm_fixed.py")
    print("2. If that works, use the conservative settings in your main script")
    print("3. Gradually increase memory utilization and other parameters")


if __name__ == "__main__":
    main()
