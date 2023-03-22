from manim import *

class RTCalc(Scene):
    def construct(self):

        # Beer Lambert law ----------------------------------------------------

        # Definition

        BeerLambert_title = Tex(r"We start from the Beer-Lambert law")
        BeerLambert_eq = MathTex(r"\frac{dI_\nu}{ds} = -\kappa_{\nu}I_\nu + \epsilon_\nu")
        VGroup(BeerLambert_title, BeerLambert_eq).arrange(DOWN)

        self.play( FadeIn(BeerLambert_title, shift=UP) )
        self.play( Write(BeerLambert_eq) )
        self.wait()

        # Rewrite

        BeerLambert_title_moved = title.copy()
        BeerLambert_eq_moved = eq.copy()
        BeerLambert_rewrite_title = Tex(r"We can rewrite it as")
        BeerLambert_rewrite_eq = MathTex(r"\frac{dI_\nu}{ds} + \kappa_{\nu}I_\nu = \epsilon_\nu")
        VGroup(BeerLambert_title_moved, BeerLambert_eq_moved, BeerLambert_rewrite_title, BeerLambert_rewrite_eq).arrange(DOWN)

        self.play(
            Transform(BeerLambert_title, BeerLambert_title_moved),
            Transform(BeerLambert_eq, BeerLambert_eq_moved),
            FadeIn(BeerLambert_rewrite_title, shift=UP)
        )
        self.play(Write(BeerLambert_rewrite_eq))
        self.wait()

        # Erase and keep rewrited form for later

        self.play(
            FadeOut(BeerLambert_title, BeerLambert_rewrite_title, BeerLambert_eq)
        )
        self.play(
            BeerLambert_rewrite_eq.animate.to_corner(UP + LEFT)
        )

        # Integrant factor ----------------------------------------------------

        # Definition

        IntegrantFractor_title = Tex(r'We introduce an "integrant factor"\\defined such as:')
        IntegrantFractor_def = MathTex(r"\frac{d\mu}{ds} = \mu \kappa_\nu")
        VGroup(new_title, new_eq).arrange(DOWN)

        self.play(
            FadeIn(IntegrantFractor_title, shift=UP),
        )
        self.play(
            Write(IntegrantFractor_def),
        )
        self.wait()

        # Solving

        IntegrantFractor_title_moved = IntegrantFractor_title.copy()
        IntegrantFractor_def_moved = IntegrantFractor_def.copy()
        IntegrantFractor_rewrite_title = Tex(r"It's form is then")
        IntegrantFractor_expression = MathTex(r"\mu = e^{\int \kappa_\nu ds}")
        VGroup(IntegrantFractor_title_moved, IntegrantFractor_def_moved, IntegrantFractor_rewrite_title, IntegrantFractor_expression).arrange(DOWN)

        self.play(
            Transform(IntegrantFractor_title, IntegrantFractor_title_moved),
            Transform(IntegrantFractor_def, IntegrantFractor_def_moved),
            FadeIn(IntegrantFractor_rewrite_title, shift=UP)
        )
        self.play(Write(IntegrantFractor_expression))
        self.wait()

        # Erase and keep expression form for later

        IntegrantFractor_expression_moved = eq2.copy().to_corner(UP + RIGHT)

        self.play(
            FadeOut(IntegrantFractor_title, IntegrantFractor_rewrite_title, IntegrantFractor_def)
        )
        self.play(
            Transform(IntegrantFractor_expression, IntegrantFractor_expression_moved)
        )

        # Beer Lambert law with integrant factor ------------------------------

        # TODO: continue the rewrite

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


    