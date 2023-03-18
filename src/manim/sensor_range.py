from manim import *

class RTCalc(Scene):
    def construct(self):

        title = Tex(r"We start from the Beer-Lambert law")
        eq = MathTex(r"\frac{dI_\nu}{ds} = -\kappa_{\nu}I_\nu + \epsilon_\nu")
        VGroup(title, eq).arrange(DOWN)

        self.play(
            FadeIn(title, shift=UP)
        )
        self.play(
            Write(eq)
        )
        self.wait()

        new_title = title.copy()
        new_eq = eq.copy()
        title2 = Tex(r"We can rewrite it as")
        eq2 = MathTex(r"\frac{dI_\nu}{ds} + \kappa_{\nu}I_\nu = \epsilon_\nu")
        VGroup(new_title, new_eq, title2, eq2).arrange(DOWN)

        self.play(
            Transform(title, new_title),
            Transform(eq, new_eq),
            FadeIn(title2, shift=UP)
        )
        self.play(Write(eq2))
        self.wait()

        eq_bottom = eq2
        new_title = Tex(r'We introduce an "integrant factor"\\defined such as:')
        new_eq = MathTex(r"\frac{d\mu}{ds} = \mu \kappa_\nu")
        VGroup(new_title, new_eq).arrange(DOWN)

        self.play(
            FadeOut(title, title2, eq)
        )
        self.play(
            eq_bottom.animate.to_corner(UP + LEFT)
        )
        self.play(
            FadeIn(new_title, shift=UP),
        )
        self.play(
            Write(new_eq),
        )
        self.wait()
        title = new_title
        eq = new_eq

        new_title = title.copy()
        new_eq = eq.copy()
        title2 = Tex(r"It's form is then")
        eq2 = MathTex(r"\mu = e^{\int \kappa_\nu ds}")
        VGroup(new_title, new_eq, title2, eq2).arrange(DOWN)

        self.play(
            Transform(title, new_title),
            Transform(eq, new_eq),
            FadeIn(title2, shift=UP)
        )
        self.play(Write(eq2))
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
        new_eq = eq.copy()
        title = Tex(r'We multiply our equation by this factor')
        VGroup(title, new_eq).arrange(DOWN)

        self.play(
            Transform(eq, new_eq)
        )
        self.play(
            FadeIn(title, shift=DOWN),
        )
        
        new_eq = MathTex(r"\mu \frac{dI_\nu}{ds} + \mu \kappa_{\nu}I_\nu = \mu \epsilon_\nu")
        VGroup(title, new_eq).arrange(DOWN)

        self.play(
            Transform(eq, new_eq)
        )
        self.wait()

        self.play(
            FadeOut(title)
        )
        self.play(
            eq.animate.to_corner(UP + LEFT),
        )
        eq_bottom = eq

        title = Tex(r"By construction, I can write that")
        eq = MathTex(r"\frac{d\mu I_\nu}{ds} = \frac{dI_\nu}{ds} \mu + \frac{d\mu}{ds} I_\nu \\ \implies \frac{d\mu I_\nu}{ds} = \frac{dI}{ds} \mu + \mu \kappa_\nu I_\nu")
        VGroup(title, eq).arrange(DOWN)

        self.play(
            FadeIn(title, shift=UP),
        )
        self.play(
            Write(eq)
        )

    