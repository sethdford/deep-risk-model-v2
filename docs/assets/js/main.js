document.addEventListener('DOMContentLoaded', function() {
  // Theme toggle functionality
  const themeToggleBtn = document.getElementById('theme-toggle-btn');
  const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
  
  // Check for saved theme preference or use the system preference
  const currentTheme = localStorage.getItem('theme') || 
                      (prefersDarkScheme.matches ? 'dark' : 'light');
  
  // Set initial theme
  if (currentTheme === 'dark') {
    document.body.classList.add('dark-theme');
  } else {
    document.body.classList.remove('dark-theme');
  }
  
  // Toggle theme when button is clicked
  themeToggleBtn.addEventListener('click', function() {
    let theme;
    
    if (document.body.classList.contains('dark-theme')) {
      document.body.classList.remove('dark-theme');
      theme = 'light';
    } else {
      document.body.classList.add('dark-theme');
      theme = 'dark';
    }
    
    // Save the preference
    localStorage.setItem('theme', theme);
  });
  
  // Search functionality
  const searchInput = document.getElementById('search-input');
  const searchResults = document.getElementById('search-results');
  let searchIndex;
  let searchData;
  
  // Load search index
  fetch('/search-data.json')
    .then(response => response.json())
    .then(data => {
      searchData = data;
      // Initialize lunr.js search index
      searchIndex = lunr(function() {
        this.ref('id');
        this.field('title', { boost: 10 });
        this.field('content');
        
        data.forEach((doc, idx) => {
          doc.id = idx;
          this.add(doc);
        });
      });
    })
    .catch(error => console.error('Error loading search data:', error));
  
  // Handle search input
  searchInput.addEventListener('input', function() {
    const query = this.value.trim();
    
    if (!query || query.length < 2) {
      searchResults.innerHTML = '';
      searchResults.style.display = 'none';
      return;
    }
    
    if (!searchIndex) {
      searchResults.innerHTML = '<div class="search-result-item">Loading search index...</div>';
      searchResults.style.display = 'block';
      return;
    }
    
    try {
      const results = searchIndex.search(query);
      
      if (results.length === 0) {
        searchResults.innerHTML = '<div class="search-result-item">No results found</div>';
      } else {
        searchResults.innerHTML = results
          .slice(0, 5) // Limit to 5 results
          .map(result => {
            const doc = searchData[result.ref];
            return `
              <a href="${doc.url}" class="search-result-item">
                <div class="search-result-title">${doc.title}</div>
                <div class="search-result-snippet">${getSnippet(doc.content, query)}</div>
              </a>
            `;
          })
          .join('');
      }
      
      searchResults.style.display = 'block';
    } catch (e) {
      console.error('Search error:', e);
      searchResults.innerHTML = '<div class="search-result-item">Search error. Try a different query.</div>';
      searchResults.style.display = 'block';
    }
  });
  
  // Hide search results when clicking outside
  document.addEventListener('click', function(event) {
    if (!searchInput.contains(event.target) && !searchResults.contains(event.target)) {
      searchResults.style.display = 'none';
    }
  });
  
  // Helper function to get a snippet of text around the query
  function getSnippet(text, query) {
    if (!text) return '';
    
    const lowerText = text.toLowerCase();
    const lowerQuery = query.toLowerCase();
    const index = lowerText.indexOf(lowerQuery);
    
    if (index === -1) return text.slice(0, 100) + '...';
    
    const start = Math.max(0, index - 40);
    const end = Math.min(text.length, index + query.length + 40);
    let snippet = text.slice(start, end);
    
    if (start > 0) snippet = '...' + snippet;
    if (end < text.length) snippet = snippet + '...';
    
    return snippet;
  }
  
  // Add active class to current page in sidebar
  const currentPath = window.location.pathname;
  const sidebarLinks = document.querySelectorAll('.sidebar-link');
  
  sidebarLinks.forEach(link => {
    if (link.getAttribute('href') === currentPath) {
      link.classList.add('active');
    }
  });
  
  // Smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      
      const targetId = this.getAttribute('href');
      if (targetId === '#') return;
      
      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        targetElement.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
        
        // Update URL hash without scrolling
        history.pushState(null, null, targetId);
      }
    });
  });
  
  // Initialize code highlighting if Prism.js is available
  if (typeof Prism !== 'undefined') {
    Prism.highlightAll();
  }
}); 