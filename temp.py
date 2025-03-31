import sys

sys.path.append("/home/shreyas/Projects/accessory_scripts/")
import slack_bot


params_fname = "temp.yaml"

slack_bot.send_slack_dm(
    f"The python process with parameter file name: '{params_fname}' completed"
)
