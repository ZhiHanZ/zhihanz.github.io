# Blog Development Guide

## Adding a New Blog Post

Create a new markdown file in `src/content/posts/`:

```bash
src/content/posts/my-new-post.md
```

### Frontmatter Template

```yaml
---
title: "Your Post Title"
date: 2025-01-10
description: "A brief description for SEO and previews"
tags: ["genai", "rust", "data-infra"]
---
```

### Required Fields
- `title` - Post title (string)
- `date` - Publication date (YYYY-MM-DD format)

### Optional Fields
- `description` - Short summary for previews and SEO
- `tags` - Array of tags for categorization
- `image` - Banner image path for social sharing (see below)

## Adding a Banner Image (Social Media Preview)

When you share a post on X (Twitter) or LinkedIn, the banner image will be displayed as the preview card.

### Step 1: Add the image to public folder

```
public/
  images/
    my-post-banner.png
```

**Recommended size:** 1200x630 pixels (optimal for Twitter/LinkedIn)

### Step 2: Reference in frontmatter

```yaml
---
title: "My Awesome Post"
date: 2025-01-10
description: "A brief description"
tags: ["genai", "rust"]
image: "/images/my-post-banner.png"
---
```

### How it works

The `image` field generates Open Graph and Twitter Card meta tags:

```html
<meta property="og:image" content="https://zhihanz.github.io/images/my-post-banner.png" />
<meta name="twitter:image" content="https://zhihanz.github.io/images/my-post-banner.png" />
<meta name="twitter:card" content="summary_large_image" />
```

### Testing Social Cards

- **Twitter/X:** https://cards-dev.twitter.com/validator
- **LinkedIn:** https://www.linkedin.com/post-inspector/
- **Facebook:** https://developers.facebook.com/tools/debug/

## Adding Images

### Option 1: Public Folder (Recommended for standalone images)

Place images in `public/images/`:

```
public/
  images/
    my-post/
      diagram.png
      screenshot.jpg
```

Reference in markdown:

```markdown
![Alt text](/images/my-post/diagram.png)
```

### Option 2: Co-located Images (For post-specific images)

Create a folder for your post:

```
src/content/posts/
  my-post/
    index.md
    hero.png
    diagram.svg
```

Reference in markdown:

```markdown
![Alt text](./hero.png)
```

## Adding a Series

For multi-part articles, create files in `src/content/series/`:

```yaml
---
title: "Building a Query Engine - Part 1"
date: 2025-01-10
description: "Introduction to query engine architecture"
tags: ["database", "rust"]
series: "building-query-engine"
part: 1
---
```

## Local Development

```bash
npm install      # Install dependencies
npm run dev      # Start dev server at localhost:4321
npm run build    # Build for production
```

## Deployment

Push to `main` branch - GitHub Actions will automatically build and deploy to GitHub Pages.

## Project Structure

```
src/
  content/
    posts/          # Blog posts (*.md)
    series/         # Multi-part series (*.md)
  pages/
    index.astro     # Home page
    about.astro     # About page
    posts/          # Post listing and detail pages
    series/         # Series listing and detail pages
    tags/           # Tag pages
public/
  images/           # Static images
  favicon.svg       # Site favicon
site.yaml           # Site configuration (title, social links, etc.)
```
