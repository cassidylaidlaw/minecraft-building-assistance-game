from distutils.core import setup, Extension
import numpy as np

setup(
    ext_modules=[
        Extension(
            "_mbag",
            sources=[
                "mbag/c_extensions/_mbagmodule.c",
                "mbag/c_extensions/action_distributions.c",
                "mbag/c_extensions/blocks.c",
                "mbag/c_extensions/mcts.c",
            ],
            include_dirs=[np.get_include()],
        )
    ],
)
