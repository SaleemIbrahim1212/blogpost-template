# frozen_string_literal: true

source "https://rubygems.org"

git_source(:github) { |repo_name| "https://github.com/#{repo_name}" }

# Core Jekyll
gem "jekyll", "~> 4.3.3"

# Jekyll plugins
gem "jekyll-gist"
gem "jekyll-sitemap"
gem "jekyll-seo-tag"
gem "jekyll-paginate"

# Windows compatibility
gem "wdm", ">= 0.1.0"

# Stdlib gems split out in Ruby 3.x (needed for Jekyll & deps)
gem "webrick", "~> 1.7"   # web server
gem "csv", "~> 3.3"       # CSV support
gem "base64"              # Base64 encoding/decoding
gem "bigdecimal"          # Arbitrary precision decimals (needed by Liquid)
gem "rexml"               # XML parsing (used by Kramdown/Jekyll)
gem "openssl"             # SSL/TLS
gem "zlib"                # Compression
