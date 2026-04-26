It is expected that all code written is extremely well documented to ease the
maintainance burden. This means that you should fill your code with
comments. All but the smallest of functions should be preceded by a
multiple-line documentation comment. Tests should make clear via comments
exactly what test they are setting up alongside the code which does it, and what
they are checking for. Where you touch uncommented code or inaccurate comments,
please add or correct the comments. Never delete the information content of
comments unless it is inaccurate.

Avoid leaving messy code for compatibility issues: freely restructure files and
modules where the impact of this is just a simple matter of renaming. When a
file gets large and seems to be diverging into more than one purpose, consider
splitting the module up.
