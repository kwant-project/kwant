{% extends "autosummary/class.rst" %}

{% block methods %}
{{ super() }}
{% if methods %}
   {% if '__call__' in members -%}
   .. automethod:: {{name}}.__call__
    {%- endif %}
{% endif %}
{% endblock %}
