"""ascii-video: render video to characters, measure how much of it survives.

The interesting claim in any ASCII art renderer is a fidelity claim, and it is almost never
measured. This package renders to character cells, renders those cells back to pixels the
way a terminal would draw them, and compares the two.
"""

__all__ = ["cast", "fidelity", "font", "player", "ramps", "raster", "render", "scenes", "study"]
__version__ = "1.0.0"
