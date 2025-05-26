

import os
import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_versions():
    """Check current package versions"""
    print("🔍 Checking Package Versions...")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import vllm
        print(f"vLLM: {vllm.__version__}")
    except ImportError:
        print("❌ vLLM not installed")
        return False
    
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("⚠️  Transformers not found")
    
    return True


def fix_pytorch_logging_issue():
    """Apply specific fix for PyTorch logging issue"""
    print("\n🔧 Applying PyTorch Logging Fix...")
    
    # Environment variables to prevent logging conflicts
    env_fixes = {
        "TORCH_LOGS": "",
        "TORCH_COMPILE_DEBUG": "0",
        "TORCHDYNAMO_VERBOSE": "0",
        "VLLM_LOGGING_LEVEL": "WARNING",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "0",  # Start with single GPU
    }
    
    for key, value in env_fixes.items():
        os.environ[key] = value
        print(f"✅ Set {key}={value}")


def create_patched_vllm_test():
    """Create a test script with patches for the logging issue"""
    
    test_script = '''#!/usr/bin/env python3
"""
Patched vLLM Test Script
Includes fixes for PyTorch logging issues in vLLM 0.8.5.post1
"""

import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Critical: Set environment variables BEFORE importing torch
os.environ["TORCH_LOGS"] = ""
os.environ["TORCH_COMPILE_DEBUG"] = "0"
os.environ["TORCHDYNAMO_VERBOSE"] = "0"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Patch torch logging before import
def patch_torch_logging():
    """Patch torch logging to prevent the get_log_level_pairs error"""
    try:
        import torch._logging._internal as torch_logging
        
        # Check if log_state is a dict (the problematic case)
        if hasattr(torch_logging, 'log_state') and isinstance(torch_logging.log_state, dict):
            print("🔧 Patching torch logging state...")
            
            # Create a simple mock object with the missing method
            class MockLogState:
                def __init__(self, original_dict):
                    self.__dict__.update(original_dict)
                
                def get_log_level_pairs(self):
                    return []
            
            # Replace the dict with our mock object
            torch_logging.log_state = MockLogState(torch_logging.log_state)
            print("✅ Torch logging patched successfully")
        
    except Exception as e:
        print(f"⚠️  Logging patch failed: {e}")

# Apply the patch
patch_torch_logging()

try:
    import torch
    print(f"✅ PyTorch {torch.__version__} imported successfully")
    
    # Apply dynamo fixes
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.verbose = False
        print("✅ PyTorch dynamo configured")
    except:
        pass
    
    # Import vLLM with error handling
    print("📦 Importing vLLM...")
    from vllm import LLM, SamplingParams
    print("✅ vLLM imported successfully")
    
    # Test with minimal configuration
    print("🧪 Testing minimal vLLM configuration...")
    
    llm = LLM(
        model="facebook/opt-125m",  # Smallest model for testing
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        max_model_len=256,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        trust_remote_code=True,
        disable_log_stats=True,
        disable_log_requests=True,
    )
    
    print("✅ vLLM engine created successfully!")
    
    # Test generation
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
        stop=["\\n"]
    )
    
    outputs = llm.generate(["Hello world, this is a"], sampling_params)
    result = outputs[0].outputs[0].text
    print(f"🎯 Test generation result: '{result}'")
    
    print("\\n🎉 SUCCESS: vLLM is working correctly!")
    print("\\n💡 You can now use these settings in your main script:")
    print("""
    llm = LLM(
        model="your-model-name",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,  # Adjust as needed
        enforce_eager=True,
        disable_custom_all_reduce=True,
        trust_remote_code=True,
        disable_log_stats=True,
        disable_log_requests=True,
    )
    """)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Try installing compatible versions:")
    print("pip install torch==2.1.2")
    print("pip install vllm==0.3.0")
    
except Exception as e:
    print(f"❌ Runtime error: {e}")
    print("\\n🛠️  Additional troubleshooting steps:")
    print("1. Restart your Python kernel/terminal")
    print("2. Check GPU memory: nvidia-smi")
    print("3. Try with CPU only: add device='cpu' to LLM()")
    print("4. Update vLLM: pip install --upgrade vllm")
'''
    
    with open("test_vllm_patched.py", "w") as f:
        f.write(test_script)
    
    print("💾 Created test_vllm_patched.py")
    return "test_vllm_patched.py"


def install_compatible_versions():
    """Install known compatible versions"""
    print("\n📥 Installing Compatible Package Versions...")
    
    # These versions are known to work well together
    install_commands = [
        "pip uninstall -y torch torchvision torchaudio",
        "pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121",
        "pip uninstall -y vllm",
        "pip install vllm==0.3.0",
        "pip install transformers==4.36.0",
        "pip install accelerate==0.25.0"
    ]
    
    for cmd in install_commands:
        print(f"\\n🔄 Running: {cmd}")
        response = input("Execute this command? (y/n/s to skip all): ").lower()
        
        if response == 's':
            print("⏭️  Skipping package installation")
            break
        elif response == 'y':
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
        else:
            print("⏭️  Skipping this command")


def create_production_script():
    """Create a production-ready script with all fixes"""
    
    prod_script = '''#!/usr/bin/env python3
"""
Production vLLM Script with PyTorch Logging Fixes
Use this as a template for your actual vLLM applications
"""

import os
import warnings

# Apply all fixes before any imports
warnings.filterwarnings("ignore")

# Environment variables for stability
os.environ.update({
    "TORCH_LOGS": "",
    "TORCH_COMPILE_DEBUG": "0", 
    "TORCHDYNAMO_VERBOSE": "0",
    "VLLM_LOGGING_LEVEL": "WARNING",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "TOKENIZERS_PARALLELISM": "false",
    "CUDA_VISIBLE_DEVICES": "0",  # Adjust for your setup
})

# Patch torch logging
def patch_torch_logging():
    try:
        import torch._logging._internal as torch_logging
        if hasattr(torch_logging, 'log_state') and isinstance(torch_logging.log_state, dict):
            class MockLogState:
                def __init__(self, original_dict):
                    self.__dict__.update(original_dict)
                def get_log_level_pairs(self):
                    return []
            torch_logging.log_state = MockLogState(torch_logging.log_state)
    except:
        pass

patch_torch_logging()

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from vllm import LLM, SamplingParams

def create_llm_engine(model_name, **kwargs):
    """Create vLLM engine with stable configuration"""
    
    default_config = {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.6,
        "max_model_len": 2048,
        "enforce_eager": True,  # Prevents compilation issues
        "disable_custom_all_reduce": True,
        "trust_remote_code": True,
        "disable_log_stats": True,
        "disable_log_requests": True,
    }
    
    # Merge user config with defaults
    config = {**default_config, **kwargs}
    
    print(f"🚀 Initializing vLLM with model: {model_name}")
    return LLM(model=model_name, **config)

def generate_text(llm, prompts, max_tokens=100, temperature=0.7):
    """Generate text with the vLLM engine"""
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["\\n\\n", "</s>"]
    )
    
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

# Example usage
if __name__ == "__main__":
    try:
        # Initialize with your model
        llm = create_llm_engine(
            "defog/sqlcoder-7b-2",  # Replace with your model
            gpu_memory_utilization=0.7,  # Adjust based on your GPU
            max_model_len=4096,
        )
        
        # Test generation
        test_prompts = [
            "CREATE TABLE users (",
            "SELECT * FROM customers WHERE"
        ]
        
        results = generate_text(llm, test_prompts, max_tokens=50)
        
        for prompt, result in zip(test_prompts, results):
            print(f"\\nPrompt: {prompt}")
            print(f"Result: {result}")
        
        print("\\n✅ vLLM is working correctly in production mode!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check the troubleshooting guide for more help.")
'''
    
    with open("vllm_production.py", "w") as f:
        f.write(prod_script)
    
    print("💾 Created vllm_production.py")
    return "vllm_production.py"


def main():
    """Main troubleshooting workflow"""
    print("🔧 vLLM PyTorch Logging Issue Fix")
    print("=" * 50)
    
    # Check current setup
    if not check_versions():
        print("❌ Please install required packages first")
        return
    
    # Apply immediate fixes
    fix_pytorch_logging_issue()
    
    # Create test script
    test_file = create_patched_vllm_test()
    
    print(f"\\n🧪 Test Script Created: {test_file}")
    print("\\n📋 Next Steps:")
    print("1. Run the test script:")
    print(f"   python {test_file}")
    print("\\n2. If test fails, try installing compatible versions")
    print("\\n3. If test succeeds, use the production script template")
    
    # Ask about package updates
    update_packages = input("\\n❓ Install compatible package versions? (y/n): ").lower()
    if update_packages == 'y':
        install_compatible_versions()
    
    # Create production template
    prod_file = create_production_script()
    print(f"\\n✅ Production template created: {prod_file}")
    
    print("\\n🎯 Summary:")
    print("- The error is caused by PyTorch logging system changes")
    print("- The patch fixes the 'get_log_level_pairs' AttributeError")
    print("- Use enforce_eager=True to avoid compilation issues")
    print("- Start with conservative GPU memory settings")


if __name__ == "__main__":
    main()
