# Deep Risk Model Documentation

This directory contains the source files for the Deep Risk Model documentation site, built with Jekyll and hosted on GitHub Pages.

## Local Development

### Prerequisites

- Ruby (version 2.7.0 or higher)
- Bundler gem

### Setup

1. Install dependencies:

```bash
cd docs
bundle install
```

2. Run the Jekyll server:

```bash
bundle exec jekyll serve
```

3. Open your browser and navigate to `http://localhost:4000`

### Converting Markdown Files

To convert existing markdown files from the repository to the GitHub Pages format:

```bash
cd docs
ruby convert_markdown.rb
```

### Generating Search Data

To generate the search data for the site:

```bash
cd docs
ruby generate_search_data.rb
```

## Directory Structure

- `_config.yml`: Jekyll configuration
- `_layouts/`: Layout templates
- `_includes/`: Reusable components
- `assets/`: Static assets (CSS, JS, images)
- `docs/`: Documentation pages
- `api-reference/`: API reference documentation
- `examples/`: Example code and usage

## Adding Content

### Adding a New Documentation Page

1. Create a new markdown file in the `docs/` directory
2. Add front matter at the top of the file:

```yaml
---
layout: default
title: Your Page Title
nav_order: 5  # Controls the order in the sidebar
---
```

3. Add your content using Markdown

### Adding an API Reference Page

1. Create a new markdown file in the `api-reference/` directory
2. Add front matter at the top of the file:

```yaml
---
layout: default
title: API Component Name
---
```

3. Document the API using Markdown

## Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to the main branch.

## Customization

### Modifying the Theme

The site uses the Just the Docs theme. To customize the theme:

1. Edit the `_config.yml` file to change theme settings
2. Modify the CSS in `assets/css/custom.scss`

### Adding JavaScript Functionality

Add custom JavaScript to `assets/js/main.js` 