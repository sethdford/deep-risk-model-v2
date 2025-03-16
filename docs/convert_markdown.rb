#!/usr/bin/env ruby
# Script to convert existing markdown files to GitHub Pages format

require 'fileutils'

# Configuration
source_dir = File.expand_path('../', __dir__)
target_dir = File.expand_path('../docs', __dir__)
docs_dir = File.join(target_dir, 'docs')
api_dir = File.join(target_dir, 'api-reference')
examples_dir = File.join(target_dir, 'examples')

# Create directories if they don't exist
[docs_dir, api_dir, examples_dir].each do |dir|
  FileUtils.mkdir_p(dir) unless Dir.exist?(dir)
end

# Mapping of source files to target locations and configurations
file_mappings = [
  {
    source: 'README.md',
    target: 'docs/index.md',
    title: 'Overview',
    nav_order: 2
  },
  {
    source: 'ARCHITECTURE.md',
    target: 'docs/architecture.md',
    title: 'Architecture',
    nav_order: 3
  },
  {
    source: 'USE_CASES.md',
    target: 'docs/use-cases.md',
    title: 'Use Cases',
    nav_order: 6
  },
  {
    source: 'examples/quantization_example.rs',
    target: 'examples/quantization.md',
    title: 'Quantization Example',
    code_type: 'rust'
  },
  {
    source: 'examples/memory_optimization_example.rs',
    target: 'examples/memory-optimization.md',
    title: 'Memory Optimization Example',
    code_type: 'rust'
  }
]

# Process each file
file_mappings.each do |mapping|
  source_path = File.join(source_dir, mapping[:source])
  target_path = File.join(target_dir, mapping[:target])
  
  if File.exist?(source_path)
    puts "Converting #{mapping[:source]} to #{mapping[:target]}"
    
    # Read source content
    content = File.read(source_path)
    
    # Create front matter
    front_matter = "---\n"
    front_matter += "layout: default\n"
    front_matter += "title: #{mapping[:title]}\n"
    front_matter += "nav_order: #{mapping[:nav_order]}\n" if mapping[:nav_order]
    front_matter += "---\n\n"
    
    # Process content based on file type
    if mapping[:source].end_with?('.rs')
      # For Rust files, create a markdown file with the code in a code block
      processed_content = "# #{mapping[:title]}\n\n"
      processed_content += "```#{mapping[:code_type]}\n#{content}\n```\n"
    else
      # For markdown files, keep the content but remove any existing front matter
      if content.start_with?('---')
        # Remove existing front matter
        content = content.sub(/---(.+?)---\s*/m, '')
      end
      processed_content = content
    end
    
    # Write to target file
    FileUtils.mkdir_p(File.dirname(target_path))
    File.write(target_path, front_matter + processed_content)
    
    puts "Successfully converted #{mapping[:source]}"
  else
    puts "Warning: Source file #{mapping[:source]} not found"
  end
end

puts "Conversion complete!" 