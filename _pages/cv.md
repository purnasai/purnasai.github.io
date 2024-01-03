---
layout: archive
title: "Paper summaries"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

<ul>{% assign sorted_talks = site.talks | sort: 'date' | reverse %}
  {% for post in sorted_talks %}
    {% include archive-single-talk-cv.html %}
    <p>{{ post.excerpt | markdownify }}</p>
    {% if post.image %}
      <img src="{{ post.image }}" alt="Image for {{ post.title }}" style="max-width: 200px;">
    {% endif %}
  {% endfor %}
</ul>

<!-- <ul>
  {% assign sorted_talks = site.talks | sort: 'date' | reverse %}
  {% for post in sorted_talks %}
    <li>
      <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
      <p>{{ post.excerpt | markdownify }}</p>
      {% if post.image %}
        <img src="{{ post.image }}" alt="Image for {{ post.title }}" style="max-width: 300px;">
      {% endif %}
    </li>
  {% endfor %}
</ul> -->
