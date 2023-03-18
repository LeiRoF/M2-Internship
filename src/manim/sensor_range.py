from manim import *
from copy import deepcopy as copy

class OpeningManim(Scene):
    def construct(self):
        title = Tex(r"This is some \LaTeX")
        basel = MathTex(r"\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}")
        VGroup(title, basel).arrange(DOWN)
        self.play(
            Write(title),
            FadeIn(basel, shift=DOWN),
        )
        self.wait()

        transform_title = Tex("That was a transform")
        transform_title.to_corner(UP + LEFT)
        self.play(
            Transform(title, transform_title),
            LaggedStart(*(FadeOut(obj, shift=DOWN) for obj in basel)),
        )
        self.wait()

        grid = NumberPlane()
        grid_title = Tex("This is a grid", font_size=72)
        grid_title.move_to(transform_title)

        self.add(grid, grid_title)  # Make sure title is on top of grid
        self.play(
            FadeOut(title),
            FadeIn(grid_title, shift=UP),
            Create(grid, run_time=3, lag_ratio=0.1),
        )
        self.wait()

        grid_transform_title = Tex(
            r"That was a non-linear function \\ applied to the grid",
        )
        grid_transform_title.move_to(grid_title, UL)
        grid.prepare_for_nonlinear_transform()
        self.play(
            grid.animate.apply_function(
                lambda p: p
                + np.array(
                    [
                        np.cos(p[1]),
                        np.sin(p[0]),
                        0,
                    ],
                ),
            ),
            run_time=3,
        )
        self.wait()
        self.play(Transform(grid_title, grid_transform_title))
        self.wait()


class RTCalc(Scene):
    def construct(self):

        title = Tex(r"We start from the following well-known equation")
        eq = MathTex(r"\frac{dI_\nu}{ds} = -\kappa_{\nu}I_\nu + \epsilon_\nu")
        VGroup(title, eq).arrange(DOWN)

        self.play(
            Write(title)
        )
        self.play(
            FadeIn(eq, shift=DOWN)
        )
        self.wait()

        new_title = title.copy()
        new_eq = eq.copy()
        title2 = Tex(r"We can rewrite it as")
        eq2 = MathTex(r"\frac{dI_\nu}{ds} + \kappa_{\nu}I_\nu = \epsilon_\nu")
        VGroup(new_title, new_eq, title2, eq2).arrange(DOWN)

        self.play(
            Transform(title, new_title),
            Transform(eq, new_eq)
        )
        self.play(Write(title2))
        self.play(FadeIn(eq2, shift=DOWN))
        self.wait()

        eq_bottom = eq2
        new_title = Tex(r'We introduce an "integrant factor"')
        new_eq = MathTex(r"\frac{d\mu}{ds} = \mu \kappa_\nu")
        VGroup(new_title, new_eq).arrange(DOWN)

        self.play(
            FadeOut(title, title2, eq),
            eq_bottom.animate.to_corner(DOWN + LEFT)
        )
        self.play(
            Write(new_title),
            FadeIn(new_eq, shift=DOWN),
        )
        self.wait()
        title = new_title
        eq = new_eq

        new_title = title.copy()
        new_eq = eq.copy()
        title2 = Tex(r'Such as')
        eq2 = MathTex(r"\mu = e^{\int \kappa_\nu ds}")
        VGroup(new_title, new_eq, title2, eq2).arrange(DOWN)

        self.play(
            Transform(title, new_title),
            Transform(eq, new_eq),
        )
        self.play(Write(title2))
        self.play(FadeIn(eq2, shift=DOWN))
        self.wait()

        eq_top = eq2.copy()
        eq_top.to_corner(UP + RIGHT)

        self.play(
            FadeOut(title, title2, eq)
        )
        self.play(
            Transform(eq2, eq_top)
        )

        eq = eq_bottom
        title = Tex(r'We add it to our equation to get')
        new_eq = MathTex(r"\mu \frac{dI_\nu}{ds} + \mu \kappa_{\nu}I_\nu = \mu \epsilon_\nu")
        VGroup(title, new_eq).arrange(DOWN)

        self.play(
            Write(title)
        )
        self.play(
            Transform(eq, new_eq)
        )
        self.wait()

