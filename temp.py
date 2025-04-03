import socket
import slack_bot

slack_bot.send_slack_dm(
        f"The python process with parameter file name:  completed on {socket.gethostname()}"
    )
