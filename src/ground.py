import logging
import asyncio
import os

from dotenv import load_dotenv

from mavsdk.asyncio import Mavsdk, Configuration, ComponentType
from mavsdk.asyncio.plugins.action import ActionAsync , ActionError
from mavsdk.asyncio.plugins.mission import MissionAsync, MissionItem, MissionPlan


logger = logging.getLogger("Ground")
logging.basicConfig(level=logging.INFO)

load_dotenv()

async def main():
    mavsdk = Mavsdk(Configuration.create_with_component_type(ComponentType.GROUND_STATION))
    await mavsdk.add_any_connection(os.getenv("PX4_ADDRESS"))
    
    _wait = os.getenv("WAIT_IN_SECONDS")
    if _wait is None:
        logger.info("The time to wait is missing")
        _wait = "10"
    drone = await mavsdk.first_autopilot()
    
    if drone is not None:
        logger.info("It's connected!!")
        await asyncio.sleep(3)
        action = ActionAsync(drone)
        try:
            await action.arm()
        except ActionError:
            logger.debug("Error while arming")
            raise

        try:
            await action.takeoff()
        except ActionError:
            logger.debug("Error while takeoff")
            raise
        
    else:
        logger.info("Failed to connect")
        
if __name__ == "__main__":
    asyncio.run(main())