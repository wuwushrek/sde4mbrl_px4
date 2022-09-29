from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

def handle_user_input(node):
    """ Handle user input via prompt
    """
    # Create a prompt
    prompt = 'Enter command >>> '
    # Invalid command
    is_last_valid = True
    # Create a prompt session
    session = PromptSession()
    # auto suggester
    auto_suggest = AutoSuggestFromHistory()
    # Create a completer with functions arguments

    # Create autocomplete based on node functions
    # Get relevant node function
    node_functions = ['arm', 'disarm', 'takeoff', 'land', 'pos', 'relpos', 'offboard']
    completer = WordCompleter(node_functions, ignore_case=True)
    while True:
        try:
            # Get user input
            if is_last_valid:
                session.prompt('')
            # Set the node in user input mode
            node.userin = True
            user_input = session.prompt(prompt, auto_suggest=auto_suggest, mouse_support=False, 
                            completer=completer, complete_while_typing=True)
            # Set the node in user input mode
            node.userin = False
            # Generate a help message with function prototype
            if user_input == 'help':
                print('Available commands:')
                print('Available functions:')
                print('arm()')
                print('disarm()')
                print('takeoff(alt)')
                print('land()')
                print('pos(x,y,z,yaw)')
                print('relpos(dx,dy,dz,dyaw)')
                print('offboard()')
                is_last_valid = True
                node.userin = False
                print('\n')
                continue
            # Parse the user input to get function and arguments
            # The user input should be like: "function_name argument1 argument2 argument3=..."
            # The function name and arguments are separated by a space
            # The arguments can be given as dict
            function_name, *args = user_input.split() if user_input else ('pos',)
            
            # Check if the function exists
            if not hasattr(node, function_name):
                print('Function {} does not exist'.format(function_name))
                is_last_valid = False
                continue
            # Get the function from the node
            function = getattr(node, function_name)
            # Separate the arguments from the values
            args_ = []
            kwargs_ = {}
            for arg in args:
                if '=' in arg:
                    key, value = arg.split('=')
                    kwargs_[key] = float(value)
                else:
                    args_.append(float(arg))
            # print(args, kwargs_, function)
            # Call the function with the arguments
            function(*args_, **kwargs_)
            is_last_valid = True
            # print('\n')
        except ValueError:
            print('Invalid input')
            is_last_valid = False
            continue
        except KeyboardInterrupt:
            # Handle keyboard interrupt
            print("KeyboardInterrupt")
            break
        except EOFError:
            # Handle end of file
            print("EOFError")
            break
        except Exception as e:
            # We dont want to exit the program if an exception occurs
            print(e)
            print('Type again\n')
            is_last_valid = False
            continue
