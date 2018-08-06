# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

"""Simple sphinx extension that allows for a section that can be
hidden/shown on click using javascript"""

from docutils import nodes
from docutils.parsers.rst import directives, Directive

id_count = 0

class specialnote(nodes.General, nodes.Element):
    pass

class SpecialNote(Directive):

    required_arguments = 1
    final_argument_whitespace = True
    has_content = True

    def run(self):
        global id_count

        self.assert_has_content()
        text = '\n'.join(self.content)

        classes = directives.class_option(self.arguments[0])

        node = specialnote(text)

        node['classes'].extend(classes)
        node['title'] = self.arguments[0]
        node['myid'] = "specialnote-id" + str(id_count)
        id_count = id_count + 1

        self.state.nested_parse(self.content, self.content_offset, node)

        return [node]

def visit_html(self, node):
    self.body.append("<div class=\"specialnote-title\" id=\"" + node['myid'] +
                     "-title\">" + node['title'])
    self.body.append("</div>")
    self.body.append("<div class=\"specialnote-body\" id=\"" +
                     node['myid'] + "-body\">")

def leave_html(self, node):
    self.body.append("</div>\n")
    #If javascript is enabled, hide the content by default. Otherwise show.
    self.body.append("<script language='javascript'>")
    self.body.append("$('" + "#" + node['myid'] +
                     "-title').append(\" <a style='float: right;' id=\\\"" +
                     node['myid'] +
                     "-button\\\" href=\\\"javascript:togglediv('" +
                     node['myid'] + "')\\\">show</a>\");")
    self.body.append("$('" + "#" + node['myid'] +
                     "-body').css('display', 'none');")
    self.body.append("</script>\n")

def visit_latex(self, node):
    self.body.append("\strong{" + node['title'] +"}\n\n")

def leave_latex(self, node):
    pass

def setup(app):
    app.add_node(specialnote,
                 html=(visit_html, leave_html),
                 latex=(visit_latex, leave_latex))

    app.add_directive('specialnote', SpecialNote)
