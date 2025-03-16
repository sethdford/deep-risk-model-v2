#!/usr/bin/env ruby
# Script to generate search data for the site

require 'json'
require 'fileutils'

# Configuration
docs_dir = File.expand_path('../docs', __dir__)
output_file = File.join(docs_dir, 'search-data.json')

# Function to extract title from front matter
def extract_title(content)
  if content.start_with?('---')
    front_matter = content.match(/---(.+?)---/m)[1]
    title_match = front_matter.match(/title:\s*(.+)/)
    return title_match[1].strip if title_match
  end
  
  # If no title in front matter, try to find first heading
  heading_match = content.match(/# (.+)/)
  return heading_match[1].strip if heading_match
  
  # Default title based on filename
  return "Untitled"
end

# Function to extract content without front matter
def extract_content(content)
  if content.start_with?('---')
    # Remove front matter
    content = content.sub(/---(.+?)---\s*/m, '')
  end
  
  # Remove code blocks to avoid cluttering search results
  content = content.gsub(/```[a-z]*\n(.+?)```/m, '')
  
  # Remove HTML tags
  content = content.gsub(/<[^>]+>/, '')
  
  # Normalize whitespace
  content = content.gsub(/\s+/, ' ').strip
  
  return content
end

# Function to generate URL from file path
def generate_url(file_path, docs_dir)
  relative_path = file_path.sub(docs_dir, '')
  # Remove .md extension and ensure leading slash
  url = relative_path.sub(/\.md$/, '')
  url = '/' + url unless url.start_with?('/')
  return url
end

# Find all markdown files
markdown_files = Dir.glob(File.join(docs_dir, '**', '*.md'))

# Generate search data
search_data = []

markdown_files.each do |file|
  begin
    content = File.read(file)
    title = extract_title(content)
    text_content = extract_content(content)
    url = generate_url(file, docs_dir)
    
    search_data << {
      title: title,
      content: text_content,
      url: url
    }
    
    puts "Processed: #{file}"
  rescue => e
    puts "Error processing #{file}: #{e.message}"
  end
end

# Write search data to JSON file
File.write(output_file, JSON.pretty_generate(search_data))

puts "Search data generated: #{output_file}"
puts "Total entries: #{search_data.size}" 