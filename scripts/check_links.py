import os
import re
import urllib.parse
import sys

def check_markdown_links(root_dir):
    md_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip node_modules and .git
        if 'node_modules' in root or '.git' in root:
            continue
        for f in files:
            if f.endswith('.md'):
                md_files.append(os.path.join(root, f))
    
    # regex to find links but NOT those preceded or followed by `
    link_pattern = re.compile(r'(?<!`)\[([^\]]+)\]\(([^)]+)\)(?!`)')
    broken_links = []

    for md_file in md_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            links = link_pattern.findall(content)
            for text, link in links:
                # Ignore web links and email links
                if link.startswith('http') or link.startswith('mailto:'):
                    continue
                
                # Split link and anchor
                parts = link.split('#')
                link_path = urllib.parse.unquote(parts[0])
                anchor = parts[1] if len(parts) > 1 else None

                # Handle anchor-only links
                if not link_path:
                    if anchor:
                        # Verify anchor exists in current file
                        if not verify_anchor(md_file, anchor):
                            broken_links.append((md_file, link, "Broken Anchor"))
                    continue

                # Resolve relative path
                target_path = os.path.normpath(os.path.join(os.path.dirname(md_file), link_path))
                
                # Check file existence
                if not os.path.exists(target_path):
                    broken_links.append((md_file, link, "Broken Path"))
                elif anchor:
                    # Verify anchor exists in target file
                    if not verify_anchor(target_path, anchor):
                        broken_links.append((md_file, link, f"Broken Anchor in {target_path}"))
    
    if broken_links:
        print("Broken links found:")
        for f, l, err in broken_links:
            print(f"File: {f} -> Link: {l} ({err})")
        return False
    else:
        print("No broken local links found.")
        return True

def generate_github_slug(text):
    """
    Simulates GitHub's anchor generation logic.
    - Lowercase
    - Remove punctuation (except hyphens and underscores)
    - Replace spaces and special chars (like &) with hyphens
    - Remove emojis
    - Collapse multiple hyphens
    """
    # Remove emojis and other non-ASCII symbols
    slug = text.encode('ascii', 'ignore').decode('ascii')
    # Lowercase
    slug = slug.lower()
    # Replace anything that isn't a letter, number, hyphen, underscore, or space with nothing
    slug = re.sub(r'[^a-z0-9\-_ ]', '', slug)
    # Replace spaces with hyphens
    slug = slug.replace(' ', '-')
    # Collapse multiple hyphens
    slug = re.sub(r'-+', '-', slug)
    # Strip leading/trailing hyphens
    slug = slug.strip('-')
    return slug

def verify_anchor(file_path, anchor):
    """Check for either <a name="..."> or GitHub-style header slug."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Check for explicit anchor (supporting variants of <a name="...">)
            # Match <a name="anchor">, <a id="anchor">, etc.
            if f'name="{anchor}"' in content or f"name='{anchor}'" in content or f'id="{anchor}"' in content:
                return True
            # Check for headers
            for line in content.split('\n'):
                if line.startswith('#'):
                    # Strip leading # and whitespace
                    header_text = line.lstrip('#').strip()
                    if generate_github_slug(header_text) == anchor:
                        return True
            return False
    except Exception:
        return False

if __name__ == "__main__":
    search_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    success = check_markdown_links(search_path)
    if not success:
        sys.exit(1)
