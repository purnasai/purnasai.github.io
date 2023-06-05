---
layout: archive
title: "Paper summaries"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

======
 <ul>{% assign sorted_talks = site.talks | sort: 'date' | reverse %}
    {% for post in sorted_talks %}
      {% include archive-single-talk-cv.html %}
    {% endfor %}
  </ul>
  
