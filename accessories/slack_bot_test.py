import socket
import slack_bot

slack_bot.send_slack_dm(
    f"This is a test message from the slack bot on {socket.gethostname()}"
)
