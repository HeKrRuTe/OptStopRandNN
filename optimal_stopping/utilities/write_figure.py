import os.path

import matplotlib.pyplot as plt


_TMPL = r"""
\begin{figure}[!hb]
\label{%(label)s}
\includegraphics[scale=1]{%(label)s}
\caption{%(caption)s}
\end{figure}
"""


def write_figure(label, caption):
  """Writes the pyplot figure and .tex file to wrap it in figure + caption.

  Args:
    label: the tex label, also name of the .pdf image and .tex file.
    caption: the caption to associate to figure.
  """
  path = os.path.abspath(os.path.join(
      os.path.dirname(__file__), "../../../latex/images", label))
  if not os.path.exists(path):
      os.makedirs(path)
  image_path = path + ".pdf"
  tex_path = path + ".tex"
  print(f"Writing {image_path} ...")
  plt.savefig(image_path)
  print(f"Writing {tex_path} ...")
  with open(tex_path, "w") as tex_f:
    tex_f.write(_TMPL % {
        "label": label,
        "caption": caption.replace("_", r"\_"),
        })
