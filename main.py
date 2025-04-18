from control import Program
from settings import model_path

def main():

    main_program = Program(model_path, max_per_line=30)
    main_program.load_models(
        gen_digits='digit_generator.ot',
        gen_letters='letter_generator.ot'
    )

    while True:
        try:
            text = input(">> ")
            if text.strip().lower() in {"exit", "quit"}:
                break
            main_program.logic(text)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    main()