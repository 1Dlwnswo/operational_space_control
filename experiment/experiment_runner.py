import argparse

def run_experiment_cli(main_func):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plot",
        dest="generate_plot",
        action="store_true",
        help="Generate plots after simulation",
    )
    parser.add_argument(
        "--no-plot",
        dest="generate_plot",
        action="store_false",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--gif",
        dest="generate_gif",
        action="store_true",
        help="Generate GIF after simulation",
    )
    parser.add_argument(
        "--no-gif",
        dest="generate_gif",
        action="store_false",
        help="Skip GIF generation",
    )

    parser.set_defaults(
        generate_plot=True,
        generate_gif=True,
    )

    args = parser.parse_args()

    print(f"Generate plot: {args.generate_plot}")
    print(f"Generate GIF: {args.generate_gif}")

    main_func(
        generate_plot=args.generate_plot,
        generate_gif=args.generate_gif,
    )