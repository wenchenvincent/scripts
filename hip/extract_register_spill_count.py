import re
import subprocess
from collections import defaultdict
import sys

def demangle_name(mangled_name):
    """Demangle the given C++ mangled name using c++filt."""
    result = subprocess.run(['c++filt', mangled_name], stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()

def process_file(file_path):
    """Process the file to extract .name and .vgpr_spill_count, and store in a dictionary."""
    with open(file_path, 'r') as file:
        content = file.read()

    # Split content into sections based on .amdgcn_target
    sections = re.split(r'\.amdgcn_target "amdgcn-amd-amdhsa--([^"]+)"', content)

    result_dict = defaultdict(dict)
    for i in range(1, len(sections), 2):
        section_name = sections[i]
        section_content = sections[i + 1]

        # Find all .name entries
        name_pattern = re.compile(r'\.name:\s+([^\s]+)')
        vgpr_spill_pattern = re.compile(r'\.vgpr_spill_count:\s+(\d+)')

        names = name_pattern.findall(section_content)
        vgpr_spill_counts = vgpr_spill_pattern.findall(section_content)

        # Ensure that there are equal number of .name and .vgpr_spill_count entries
        if len(names) != len(vgpr_spill_counts):
            raise ValueError(f"Mismatch between number of names and vgpr_spill_count entries in section {section_name}.")

        for name, vgpr_spill_count in zip(names, vgpr_spill_counts):
            demangled_name = demangle_name(name)
            result_dict[section_name][demangled_name] = int(vgpr_spill_count)

    return result_dict

# Example usage
file_path = sys.argv[1]
result = process_file(file_path)
for section, names in result.items():
    print(f"Section: {section}")
    for name, vgpr_spill_count in names.items():
        if int(vgpr_spill_count) > 0:
            print(f'  {name}: {vgpr_spill_count}')
