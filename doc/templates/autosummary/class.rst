{% extends "!autosummary/class.rst" %}

{% block methods %}
{% if methods %}
   .. rubric:: Methods
   {% for item in methods %}
      {%- if not item.startswith('_') or item in ['__call__'] %}
   .. automethod:: {{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
   .. rubric:: Attributes
   {% for item in attributes %}
      {%- if not item.startswith('_') %}
   .. autoattribute:: {{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
{% endif %}
{% endblock %}
