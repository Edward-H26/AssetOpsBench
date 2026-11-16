"""The one long-lived service thread of a metered server process (CausalDiagnosis/docs/environment.md, the clock of the thinking
time).

Every call of the meter runs on this thread's own event loop, handed over to it from the stdio loop, so the thread
time of this thread between the entry and the end of a call is the work of that call alone: the stdio loop of the process
parses requests on the main thread and never on this one. The store's loop-bound clients, the meter's lock and the clock
patch of the meter then all live on one loop and one thread. A cancelled caller cancels its call on the service loop."""
import asyncio
import concurrent.futures
import contextvars
import threading


class ServiceLoop:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._serve, name="service", daemon=True)
        self.thread.start()

    def _serve(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def run(self, coroutine):
        """Run a coroutine on the service loop in a copy of the caller's context, so that what the transcript wrapper keeps
        for the call being served reaches the call (transcript.note), and await its result on the caller's loop. A cancelled
        caller cancels the task on the service loop."""
        context = contextvars.copy_context()
        result = concurrent.futures.Future()

        def start():
            task = self.loop.create_task(coroutine, context=context)

            def finished(done):
                if done.cancelled():
                    result.cancel()
                elif done.exception() is not None:
                    result.set_exception(done.exception())
                else:
                    result.set_result(done.result())

            task.add_done_callback(finished)
            result.add_done_callback(lambda future: future.cancelled() and self.loop.call_soon_threadsafe(task.cancel))

        self.loop.call_soon_threadsafe(start)
        return await asyncio.wrap_future(result)
