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

        eq_left = eq2
        new_title = Tex(r'We introduce an "integrant factor"\\defined such as:')
        new_eq = MathTex(r"\frac{d\mu}{ds} = \mu \kappa_\nu")
        VGroup(new_title, new_eq).arrange(DOWN)

        self.play(
            FadeOut(title, title2, eq)
        )
        self.play(
            eq_left.animate.to_corner(UP + LEFT)
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

        eq_right = eq2.copy()
        eq_right.to_corner(UP + RIGHT)

        self.play(
            FadeOut(title, title2, eq)
        )
        self.play(
            Transform(eq2, eq_right)
        )

        eq = eq_left
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
        eq_left = eq

        title = Tex(r"By construction, I can write that")
        eq = MathTex(r"\frac{d\mu I_\nu}{ds} = \frac{dI_\nu}{ds} \mu + \frac{d\mu}{ds} I_\nu")
        imply = MathTex(r"\implies")
        eq2 = MathTex(r"\frac{d\mu I_\nu}{ds} = \frac{dI}{ds} \mu + \mu \kappa_\nu I_\nu")
        imply_group = VGroup(imply, eq2).arrange(RIGHT)
        VGroup(title, eq, imply_group).arrange(DOWN)

        self.play(
            FadeIn(title, shift=UP),
        )
        self.play(
            Write(eq)
        )
        self.play(
            Write(imply),
            Write(eq2)
        )



        

        new_title = Tex("Let's inject that into the previous equation to simplify it")
        new_eq = eq_left.copy()

        new_eq2 = eq2.copy()

        VGroup(new_title, new_eq, new_eq2).arrange(DOWN)

        self.play(
            FadeOut(title, eq, imply),
            Transform(eq2, new_eq2),
            Transform(eq_left, new_eq)
        )

        eq = eq_left
        VGroup(new_title, eq, new_eq2)

        self.play(
            FadeIn(new_title, shift=DOWN)
        )
        title = new_title

        new_eq = MathTex(r"\frac{d\mu I_\nu}{ds} = \mu \epsilon_\nu")

        self.play(
            FadeOut(eq2, shift=UP),
            Transform(eq, new_eq)
        )

        self.wait()

        new_title = title.copy()
        new_eq = eq.copy()
        eq2 = MathTex("\implies d(\mu I_\nu) = \epsilon_\nu \mu ds")
        VGroup(new_title, new_eq, eq2).arrange(DOWN)
        self.play(
            Transform(title, new_title),
            Transform(eq, new_eq),
            Write(eq2)
        )
        self.wait()

        new_title = title.copy()
        new_eq = eq.copy()
        new_eq2 = MathTex("\implies d(\mu I_\nu) = \epsilon_\nu e^{\int \kappa_\nu ds} ds")
        eq_right_copy = eq_right.copy()

        VGroup(new_title, new_eq, new_eq2).arrange(DOWN)
        self.play(
            Transform(title, new_title),
            Transform(eq, new_eq),
            FadeOut(eq_right_copy, shift=DOWN + LEFT),
            Transform(eq2, new_eq2)
        )
        self.wait()

        new_title = eq.copy()
        new_eq = eq2.copy()
        new_eq2 = MathTex(r"\implies \mu I_\nu = \int \epsilon_\nu e^{\int \kappa_\nu ds'} ds + \mu(0)I_\nu(0)")
        VGroup(new_title, new_eq, new_eq2).arrange(DOWN)

        self.play(
            FadeOut(title, shift=UP),
            Transform(eq, new_title),
            Transform(eq2, new_eq),
        )
        self.play(
            Write(new_eq2)
        )
        self.wait()


    