import alluvial
import matplotlib.pyplot as plt
import numpy as np

def test_alluvial_general_example1():
    # TODO: Turn input_data into fixture (example1_input_data)? 
    try:
        input_data = {'a': {'aa': 0.3, 'cc': 0.7,},
                    'b': {'aa': 2, 'bb': 0.5,},
                    'c': {'aa': 0.5, 'bb': 0.5, 'cc': 1.5,}}

        ax = alluvial.plot(input_data)
        fig = ax.get_figure()
        fig.set_size_inches(5,5)

        assert True
    except:
        assert False

def test_alluvial_general_example2():
    try:
        import alluvial
        import matplotlib.pyplot as plt
        import matplotlib.cm
        import numpy as np

        # Generating the input_data:
        seed=7
        np.random.seed(seed)
        def rand_letter(num): return chr(ord('A')+int(num*np.random.rand()))

        input_data = [[rand_letter(15), rand_letter(5)*2] for _ in range(50)]

        # Plotting:
        cmap = matplotlib.cm.get_cmap('jet')
        ax = alluvial.plot(
            input_data,  alpha=0.4, color_side=1, rand_seed=seed, figsize=(7,5),
            disp_width=True, wdisp_sep=' '*2, cmap=cmap, fontname='Monospace',
            labels=('Capitals', 'Double Capitals'), label_shift=2)
        ax.set_title('Utility display', fontsize=14, fontname='Monospace')

        assert True
    except:
        assert False

