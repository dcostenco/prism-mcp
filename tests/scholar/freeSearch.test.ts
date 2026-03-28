import { describe, it, expect, vi, beforeEach } from "vitest";
import { searchYahooFree, scrapeArticleLocal } from "../../src/scholar/freeSearch.js";

// Mock the global fetch API
const originalFetch = global.fetch;

describe("Free Local Search & Scraping (Zero-Config Fallback)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  describe("searchYahooFree", () => {
    it("should parse standard Yahoo Search HTML results and extract links", async () => {
      // Create a mock Yahoo Search HTML response using the `.algo` class
      const mockYahooHtml = `
        <html>
          <body>
            <div class="algo">
              <h3 class="title"><a href="https://example.com/result1">Result 1 Title</a></h3>
              <div class="compText">Snippet 1 text</div>
            </div>
            <div class="algo">
              <h3 class="title"><a href="https://example.com/result2">Result 2 Title</a></h3>
              <div class="compText">Snippet 2 text</div>
            </div>
          </body>
        </html>
      `;

      vi.spyOn(global, "fetch").mockResolvedValueOnce(
        new Response(mockYahooHtml, { status: 200, statusText: "OK" })
      );

      const results = await searchYahooFree("test query", 2);
      
      expect(global.fetch).toHaveBeenCalledTimes(1);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining("https://search.yahoo.com/search?p=test%20query"),
        expect.any(Object)
      );

      expect(results.length).toBe(2);
      expect(results[0].url).toBe("https://example.com/result1");
      expect(results[0].title).toBe("Result 1 Title");
      expect(results[1].url).toBe("https://example.com/result2");
    });

    it("should handle Yahoo redirect URLs and extract the clean destination URL", async () => {
      // Yahoo often wraps URLs in redirects like:
      // https://r.search.yahoo.com/_ylt=.../RU=https://real-site.com/article/RK=...
      const mockYahooHtml = `
        <html>
          <body>
            <div class="algo">
              <h3 class="title"><a href="https://r.search.yahoo.com/_ylt=abc/RU=https://real-site.com/article/RK=0">Redirect Title</a></h3>
              <div class="compText">Snippet 1</div>
            </div>
            <div class="algo">
              <h3 class="title"><a href="https://r.search.yahoo.com/_ylt=def/RU=http%3A%2F%2Fencoded-site.com%2Fpath/RK=0">Encoded Title</a></h3>
              <div class="compText">Snippet 2</div>
            </div>
          </body>
        </html>
      `;

      vi.spyOn(global, "fetch").mockResolvedValueOnce(
        new Response(mockYahooHtml, { status: 200, statusText: "OK" })
      );

      const results = await searchYahooFree("redirect test", 2);
      
      expect(results.length).toBe(2);
      // The logic should extract the 'RU=' part and decode it
      expect(results[0].url).toBe("https://real-site.com/article");
      expect(results[1].url).toBe("http://encoded-site.com/path");
    });

    it("should elegantly handle empty search results", async () => {
      // HTML without any matching class names
      const mockYahooHtml = `<html><body><div>No results found for your query.</div></body></html>`;

      vi.spyOn(global, "fetch").mockResolvedValueOnce(
        new Response(mockYahooHtml, { status: 200, statusText: "OK" })
      );

      const results = await searchYahooFree("empty query", 3);
      
      expect(results.length).toBe(0);
    });

    it("should throw an error if Yahoo Search fails (e.g. 429 Too Many Requests)", async () => {
      vi.spyOn(global, "fetch").mockResolvedValueOnce(
        new Response("Too Many Requests", { status: 429, statusText: "Too Many Requests" })
      );

      await expect(searchYahooFree("blocked query", 3)).rejects.toThrow("Yahoo Search failed with status: 429");
    });

    it("should handle malformed Yahoo Search HTML without crashing", async () => {
      // If Yahoo changes their HTML structure significantly, it should return an empty array, not crash.
      const malformedHtml = `<html><body><div class="algo">just some text no tags</div></body></html>`;

      vi.spyOn(global, "fetch").mockResolvedValueOnce(
        new Response(malformedHtml, { status: 200, statusText: "OK" })
      );

      const results = await searchYahooFree("malformed query", 3);
      
      expect(results.length).toBe(0);
    });

    it("should correctly limit the number of search results returned", async () => {
      const mockYahooHtml = `
        <html>
          <body>
            <div class="algo"><h3 class="title"><a href="https://example.com/1">Result 1</a></h3></div>
            <div class="algo"><h3 class="title"><a href="https://example.com/2">Result 2</a></h3></div>
            <div class="algo"><h3 class="title"><a href="https://example.com/3">Result 3</a></h3></div>
            <div class="algo"><h3 class="title"><a href="https://example.com/4">Result 4</a></h3></div>
          </body>
        </html>
      `;

      vi.spyOn(global, "fetch").mockResolvedValueOnce(
        new Response(mockYahooHtml, { status: 200, statusText: "OK" })
      );

      // Request only LIMIT=2
      const results = await searchYahooFree("limit query", 2);
      
      expect(results.length).toBe(2);
      expect(results[0].url).toBe("https://example.com/1");
      expect(results[1].url).toBe("https://example.com/2");
    });
  });

  describe("scrapeArticleLocal", () => {
    it("should fetch an article, parse with Readability, and output clean Markdown", async () => {
      // A mock HTML payload simulating an article
      const mockArticleHtml = `
        <!DOCTYPE html>
        <html>
          <head>
            <title>My Awesome Article</title>
          </head>
          <body>
            <nav>This is a menu, it should be ignored by Readability.</nav>
            <article>
              <h1>Main Article Heading</h1>
              <p>This is the first paragraph with some <strong>bold</strong> text.</p>
              <div class="ads">Buy our product!</div>
              <p>This is the second paragraph.</p>
            </article>
            <footer>Copyright 2026</footer>
          </body>
        </html>
      `;

      vi.spyOn(global, "fetch").mockResolvedValueOnce(
        new Response(mockArticleHtml, { status: 200, headers: { "Content-Type": "text/html" } })
      );

      const scraped = await scrapeArticleLocal("https://example.com/article");

      // Verify the fetch
      expect(global.fetch).toHaveBeenCalledTimes(1);
      expect(global.fetch).toHaveBeenCalledWith("https://example.com/article", expect.any(Object));

      // Verify the parsed output fields
      expect(scraped.title).toBe("My Awesome Article");
      
      // Turndown should convert the header and paragraphs to Markdown
      expect(scraped.content).toContain("# Main Article Heading");
      expect(scraped.content).toContain("This is the first paragraph with some **bold** text.");
      expect(scraped.content).toContain("This is the second paragraph.");
      
      // Readability should strip out nav and footer, Turndown shouldn't see them
      expect(scraped.content).not.toContain("This is a menu");
      expect(scraped.content).not.toContain("Copyright");
    });

    it("should throw an error if the URL cannot be fetched (e.g., 403 Forbidden cloudflare block)", async () => {
      // Readability/JSDOM implementation uses response.statusText
      vi.spyOn(global, "fetch").mockResolvedValueOnce(
        new Response("Forbidden", { status: 403, statusText: "Forbidden" })
      );

      await expect(scrapeArticleLocal("https://blocked.com")).rejects.toThrow("Failed to fetch article HTML: Forbidden");
    });

    it("should throw an error if Readability cannot parse the main content", async () => {
      // An empty document with no readable content
      const emptyHtml = `<html><head><title>Empty</title></head><body></body></html>`;

      vi.spyOn(global, "fetch").mockResolvedValueOnce(
        new Response(emptyHtml, { status: 200 })
      );

      await expect(scrapeArticleLocal("https://empty.com")).rejects.toThrow("Readability could not parse the article content.");
    });

    it("should handle articles with missing titles gracefully, reverting to Unknown Title", async () => {
      const noTitleHtml = `
        <!DOCTYPE html>
        <html>
          <head></head>
          <body>
            <article>
              <p>This article has plenty of content but absolutely no title tags or headings!</p>
              <p>Just some paragraphs for readability to latch onto.</p>
            </article>
          </body>
        </html>
      `;

      vi.spyOn(global, "fetch").mockResolvedValueOnce(
        new Response(noTitleHtml, { status: 200, headers: { "Content-Type": "text/html" } })
      );

      const scraped = await scrapeArticleLocal("https://notitle.com/article");

      expect(scraped.title).toBe("Unknown Title");
      expect(scraped.content).toContain("This article has plenty of content");
    });
  });
});
