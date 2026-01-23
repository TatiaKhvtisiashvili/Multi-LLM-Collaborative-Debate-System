# test_config_quick.py
import yaml
import sys


def test_config():
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Config loaded successfully!")

        # Check structure
        required_sections = ['models', 'debate_settings', 'evaluation']
        for section in required_sections:
            if section in config:
                print(f"✅ Found '{section}' section")
            else:
                print(f"❌ Missing '{section}' section")
                return False

        # Check models
        models = config['models']
        print(f"\nFound {len(models)} models:")
        for name, details in models.items():
            print(f"  - {name}: {details.get('provider')}/{details.get('model_name')}")

        return True

    except yaml.YAMLError as e:
        print(f"❌ YAML Error: {e}")
        print("\nTrying to load with alternative method...")
        return try_alternative_load()
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def try_alternative_load():
    """Try alternative loading methods"""
    print("\nAttempting alternative fixes...")

    # Method 1: Try to clean the file
    try:
        with open('config.yaml', 'r') as f:
            lines = f.readlines()

        # Remove tabs
        cleaned = []
        for i, line in enumerate(lines):
            if '\t' in line:
                print(f"  Found tab at line {i + 1}, replacing with spaces")
                line = line.replace('\t', '    ')
            cleaned.append(line)

        # Try to parse cleaned content
        config = yaml.safe_load(''.join(cleaned))
        print("✅ Successfully loaded after removing tabs!")
        return True

    except Exception as e:
        print(f"❌ Alternative method failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing config.yaml...")
    if test_config():
        print("\n" + "=" * 50)
        print("✅ Config is valid!")
        sys.exit(0)
    else:
        print("\n" + "=" * 50)
        print("❌ Config has issues. Creating a fresh one...")

        # Create fresh config
        import create_clean_config  # Run the script above

        print("\n✅ New config created. Please try again.")
        sys.exit(1)