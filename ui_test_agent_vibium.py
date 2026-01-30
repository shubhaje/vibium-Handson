"""
Multi-Agent UI Testing System with Vibium
Uses Ollama LLM with Planner, Executor, and Validator agents
Integrates with Vibium for browser automation
"""

import json
import asyncio
import socket
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
import time


@dataclass
class TestStep:
    """Represents a single test step"""
    action: str
    target: str
    value: Optional[str] = None
    expected_result: Optional[str] = None
    step_number: int = 0


@dataclass
class TestPlan:
    """Represents a complete test plan"""
    objective: str
    steps: List[TestStep]
    success_criteria: List[str]


@dataclass
class ExecutionResult:
    """Results from executing a test step"""
    step_number: int
    action: str
    success: bool
    message: str
    screenshot_path: Optional[str] = None
    actual_result: Optional[str] = None


class OllamaClient:
    """Client for interacting with Ollama LLM"""
    
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        """Generate response from Ollama"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return ""


class PlannerAgent:
    """Agent responsible for creating test plans"""
    
    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
        self.system_prompt = """You are an expert UI test planner. Your job is to create detailed, 
        step-by-step test plans for web applications. 
        
        Output your test plan in the following JSON format:
        {
            "objective": "Clear description of what we're testing",
            "steps": [
                {
                    "step_number": 1,
                    "action": "navigate",
                    "target": "https://example.com",
                    "value": "",
                    "expected_result": "what should happen"
                }
            ],
            "success_criteria": ["criterion 1", "criterion 2"]
        }
        
        CRITICAL RULES:
        - ALWAYS use double quotes (") for strings, never single quotes
        - NEVER use trailing commas before closing brackets ] or braces }
        - For 'navigate' actions, ALWAYS use full URLs (https://...)
        - For 'click', 'type', 'verify' actions, use CSS selectors
        - Keep value as empty string "" if not needed, don't omit it
        - Return ONLY valid JSON with no trailing commas"""
    
    def create_plan(self, user_prompt: str) -> TestPlan:
        """Create a test plan based on user prompt"""
        prompt = f"""Create a detailed UI test plan for the following requirement:

{user_prompt}

IMPORTANT JSON RULES:
1. Use ONLY double quotes for all strings
2. NO trailing commas before ] or }}
3. For 'navigate' actions, use full URLs (https://...)
4. Always include "value" field (use "" if empty)
5. Return ONLY the JSON object, no markdown, no backticks, no explanations

Test Plan:"""

        response = self.llm.generate(prompt, self.system_prompt, temperature=0.3)
        
        print(f"\n[DEBUG] Raw LLM Response:\n{response}\n")

        try:
            json_str = self._extract_json(response)
            print(f"\n[DEBUG] Extracted JSON:\n{json_str}\n")
            
            plan_data = json.loads(json_str)

            # Convert to TestPlan object
            steps = []
            for idx, step in enumerate(plan_data.get('steps', [])):
                # Ensure all required fields exist
                step_obj = TestStep(
                    step_number=int(step.get('step_number', idx + 1)),
                    action=str(step.get('action', 'navigate')),
                    target=str(step.get('target', '')),
                    value=str(step.get('value', '') or ''),
                    expected_result=str(step.get('expected_result', '') or '')
                )
                steps.append(step_obj)

            return TestPlan(
                objective=plan_data.get('objective', user_prompt),
                steps=steps,
                success_criteria=plan_data.get('success_criteria', [])
            )
        except Exception as e:
            print(f"[ERROR] Failed to parse JSON response: {e}")
            print(f"[ERROR] Response was:\n{response}\n")
            # Return a basic plan if parsing fails
            return TestPlan(
                objective=user_prompt,
                steps=[TestStep(
                    step_number=1, 
                    action="navigate", 
                    target="https://www.google.com", 
                    value="",
                    expected_result="Page loads successfully"
                )],
                success_criteria=["Test completes without errors"]
            )

    def _extract_json(self, text: str) -> str:
        """Extract and clean JSON from LLM response"""
        if not text:
            raise ValueError("Empty response")

        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Find JSON object boundaries
        start = text.find('{')
        end = text.rfind('}')
        
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in response")
        
        json_str = text[start:end+1]
        
        # Remove comments
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Fix trailing commas - MORE AGGRESSIVE
        # Remove comma before closing brace or bracket with optional whitespace/newlines
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Fix multiple trailing commas
        json_str = re.sub(r',+\s*([}\]])', r'\1', json_str)
        
        # Fix quotes (normalize smart quotes)
        json_str = json_str.replace('"', '"').replace('"', '"')
        json_str = json_str.replace(''', '"').replace(''', '"')
        
        # Remove any whitespace-only lines
        lines = json_str.split('\n')
        lines = [line for line in lines if line.strip()]
        json_str = '\n'.join(lines)
        
        return json_str


class VibiumExecutorAgent:
    """Agent responsible for executing test steps using Vibium"""
    
    def __init__(self, llm_client: OllamaClient, headless: bool = False):
        self.llm = llm_client
        self.browser = None
        self.page = None
        self.headless = headless
        self.wait_timeout = 10000  # milliseconds
        
    async def initialize_browser(self):
        """Initialize the Vibium browser"""
        try:
            # Try multiple import styles to support different vibium versions
            try:
                from vibium import Browser as VibiumBrowser
            except Exception:
                try:
                    from vibium import browser as VibiumBrowser
                except Exception:
                    import vibium as _vib
                    VibiumBrowser = getattr(_vib, 'Browser', None) or getattr(_vib, 'browser', None)
                    if VibiumBrowser is None:
                        raise ImportError("Could not find 'Browser' or 'browser' in vibium package")

            # Instantiate browser
            if callable(VibiumBrowser):
                try:
                    self.browser = VibiumBrowser(headless=self.headless)
                except TypeError:
                    self.browser = VibiumBrowser()
            else:
                class_candidate = getattr(VibiumBrowser, 'Browser', None)
                if class_candidate:
                    self.browser = class_candidate(headless=self.headless)
                else:
                    raise ImportError("vibium browser interface not compatible")

            # Launch browser with port handling
            if hasattr(self.browser, 'launch'):
                launch = getattr(self.browser, 'launch')
                launched = None

                def _find_free_port() -> int:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.bind(("", 0))
                    port = s.getsockname()[1]
                    s.close()
                    return port

                ports_to_try = [_find_free_port() for _ in range(5)]

                chrome_paths = [
                    os.path.join(os.environ.get('PROGRAMFILES', "C:\\Program Files"), 'Google', 'Chrome', 'Application', 'chrome.exe'),
                    os.path.join(os.environ.get('PROGRAMFILES(X86)', "C:\\Program Files (x86)"), 'Google', 'Chrome', 'Application', 'chrome.exe')
                ]
                executable_path = None
                for p in chrome_paths:
                    if p and os.path.exists(p):
                        executable_path = p
                        break

                for port in ports_to_try:
                    try:
                        kwargs = {'port': port}
                        if executable_path:
                            kwargs['executable_path'] = executable_path

                        if asyncio.iscoroutinefunction(launch):
                            try:
                                launched = await launch(headless=self.headless, **kwargs)
                            except TypeError:
                                launched = await launch(**kwargs)
                        else:
                            try:
                                launched = launch(headless=self.headless, **kwargs)
                            except TypeError:
                                launched = launch(**kwargs)

                        if launched:
                            print(f"✓ Browser launched (port {port})")
                            break
                    except Exception as e:
                        print(f"[DEBUG] Launch attempt failed (port={port}): {e}")
                        await asyncio.sleep(0.2)

                if launched:
                    self.browser = launched
                else:
                    raise Exception("Could not launch browser on any port")

            # Start the browser
            start = getattr(self.browser, 'start', None)
            if start:
                if asyncio.iscoroutinefunction(start):
                    await self.browser.start()
                elif callable(start):
                    self.browser.start()

            # Create a new page
            if hasattr(self.browser, 'new_page'):
                new_page = getattr(self.browser, 'new_page')
                if asyncio.iscoroutinefunction(new_page):
                    self.page = await new_page()
                else:
                    self.page = new_page()
            elif hasattr(self.browser, 'newPage'):
                np = getattr(self.browser, 'newPage')
                if asyncio.iscoroutinefunction(np):
                    self.page = await np()
                else:
                    self.page = np()
            elif hasattr(self.browser, 'page'):
                self.page = getattr(self.browser, 'page')
            else:
                # Create adapter for browser-as-page objects
                if hasattr(self.browser, 'goto') or hasattr(self.browser, 'go'):
                    class VibiumPageAdapter:
                        def __init__(self, vibe):
                            self._vibe = vibe
                            self.url = None

                        async def goto(self, url, timeout=None):
                            func = getattr(self._vibe, 'goto', None) or getattr(self._vibe, 'go', None)
                            if asyncio.iscoroutinefunction(func):
                                await func(url)
                            else:
                                func(url)
                            self.url = url

                        async def query_selector(self, selector):
                            finder = getattr(self._vibe, 'query_selector', None) or getattr(self._vibe, 'find', None)
                            if finder is None:
                                return None
                            if asyncio.iscoroutinefunction(finder):
                                return await finder(selector)
                            else:
                                return finder(selector)

                        async def wait_for_selector(self, selector, timeout=None):
                            finder = getattr(self._vibe, 'find', None)
                            if finder is None:
                                raise Exception('wait_for_selector not supported')
                            for _ in range(10):
                                result = finder(selector) if not asyncio.iscoroutinefunction(finder) else await finder(selector)
                                if result:
                                    return result
                                await asyncio.sleep(0.5)
                            raise Exception(f'Timeout waiting for selector: {selector}')

                    self.page = VibiumPageAdapter(self.browser)
                else:
                    raise Exception("Could not create a new page from vibium browser")

            print("✓ Vibium browser initialized")

        except ImportError:
            print("ERROR: Vibium not installed. Install with: pip install vibium")
            raise
        except Exception as e:
            print(f"ERROR initializing Vibium: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def close_browser(self):
        """Close the browser"""
        if self.browser:
            try:
                closed = False
                for name in ('close', 'stop', 'shutdown'):
                    method = getattr(self.browser, name, None)
                    if method:
                        if asyncio.iscoroutinefunction(method):
                            await method()
                        else:
                            method()
                        closed = True
                        break

                self.browser = None
                self.page = None
            except Exception as e:
                print(f"Error closing browser: {e}")
    
    async def execute_step(self, step: TestStep) -> ExecutionResult:
        """Execute a single test step"""
        if not self.browser:
            await self.initialize_browser()
        
        try:
            if step.action == "navigate":
                return await self._execute_navigate(step)
            elif step.action == "click":
                return await self._execute_click(step)
            elif step.action == "type":
                return await self._execute_type(step)
            elif step.action == "verify":
                return await self._execute_verify(step)
            elif step.action == "wait":
                return await self._execute_wait(step)
            elif step.action == "scroll":
                return await self._execute_scroll(step)
            else:
                return ExecutionResult(
                    step_number=step.step_number,
                    action=step.action,
                    success=False,
                    message=f"Unknown action: {step.action}"
                )
        except Exception as e:
            return ExecutionResult(
                step_number=step.step_number,
                action=step.action,
                success=False,
                message=f"Error executing step: {str(e)}"
            )
    
    async def _execute_navigate(self, step: TestStep) -> ExecutionResult:
        """Navigate to a URL"""
        url = step.target if step.target.startswith('http') else f"https://{step.target}"
        
        try:
            await self.page.goto(url, timeout=30000)
            await asyncio.sleep(2)
            
            return ExecutionResult(
                step_number=step.step_number,
                action=step.action,
                success=True,
                message=f"Navigated to {url}",
                actual_result=getattr(self.page, 'url', url)
            )
        except Exception as e:
            return ExecutionResult(
                step_number=step.step_number,
                action=step.action,
                success=False,
                message=f"Navigation failed: {str(e)}"
            )
    
    async def _find_element(self, target: str):
        """Find element using Vibium selector"""
        try:
            # Try as CSS selector first
            element = await self.page.query_selector(target)
            if element:
                return element
            
            # Try finding by text
            element = await self.page.query_selector(f"text={target}")
            if element:
                return element
            
            # Try aria label
            element = await self.page.query_selector(f"[aria-label*='{target}']")
            if element:
                return element
            
            raise Exception(f"Could not find element: {target}")
            
        except Exception as e:
            raise Exception(f"Element not found: {target} - {str(e)}")
    
    async def _execute_click(self, step: TestStep) -> ExecutionResult:
        """Click an element"""
        try:
            element = await self._find_element(step.target)
            await element.click()
            await asyncio.sleep(1)
            
            return ExecutionResult(
                step_number=step.step_number,
                action=step.action,
                success=True,
                message=f"Clicked element: {step.target}"
            )
        except Exception as e:
            return ExecutionResult(
                step_number=step.step_number,
                action=step.action,
                success=False,
                message=f"Click failed: {str(e)}"
            )
    
    async def _execute_type(self, step: TestStep) -> ExecutionResult:
        """Type text into an element"""
        try:
            element = await self._find_element(step.target)
            
            # Clear if element has clear method
            if hasattr(element, 'clear'):
                await element.clear()
            
            # Type the value
            if hasattr(element, 'type'):
                await element.type(step.value or "")
            elif hasattr(element, 'fill'):
                await element.fill(step.value or "")
            
            # Press Enter if specified
            if step.value and "ENTER" in step.value.upper():
                if hasattr(self.page, 'keyboard'):
                    await self.page.keyboard.press("Enter")
            
            return ExecutionResult(
                step_number=step.step_number,
                action=step.action,
                success=True,
                message=f"Typed '{step.value}' into {step.target}"
            )
        except Exception as e:
            return ExecutionResult(
                step_number=step.step_number,
                action=step.action,
                success=False,
                message=f"Type failed: {str(e)}"
            )
    
    async def _execute_verify(self, step: TestStep) -> ExecutionResult:
        """Verify element exists or has expected content"""
        try:
            element = await self._find_element(step.target)
            
            if step.value:
                # Verify element contains expected text
                if hasattr(element, 'text_content'):
                    element_text = await element.text_content()
                elif hasattr(element, 'inner_text'):
                    element_text = await element.inner_text()
                else:
                    element_text = str(element)
                
                success = step.value.lower() in element_text.lower()
                message = f"Verified '{step.value}' in element" if success else f"Expected '{step.value}' not found in '{element_text}'"
            else:
                # Just verify element exists
                success = True
                message = f"Verified element exists: {step.target}"
            
            return ExecutionResult(
                step_number=step.step_number,
                action=step.action,
                success=success,
                message=message,
                actual_result=element_text if step.value else None
            )
        except Exception as e:
            return ExecutionResult(
                step_number=step.step_number,
                action=step.action,
                success=False,
                message=f"Verification failed: {str(e)}"
            )
    
    async def _execute_wait(self, step: TestStep) -> ExecutionResult:
        """Wait for specified seconds or for element"""
        try:
            if step.value and step.value.isdigit():
                seconds = int(step.value)
                await asyncio.sleep(seconds)
                message = f"Waited {seconds} seconds"
            else:
                # Wait for element to appear
                await self.page.wait_for_selector(step.target, timeout=self.wait_timeout)
                message = f"Element appeared: {step.target}"
            
            return ExecutionResult(
                step_number=step.step_number,
                action=step.action,
                success=True,
                message=message
            )
        except Exception as e:
            return ExecutionResult(
                step_number=step.step_number,
                action=step.action,
                success=False,
                message=f"Wait failed: {str(e)}"
            )
    
    async def _execute_scroll(self, step: TestStep) -> ExecutionResult:
        """Scroll the page"""
        try:
            if step.value:
                # Scroll specific amount
                if hasattr(self.page, 'evaluate'):
                    await self.page.evaluate(f"window.scrollBy(0, {step.value})")
                message = f"Scrolled {step.value}px"
            else:
                # Scroll to element
                element = await self._find_element(step.target)
                if hasattr(element, 'scroll_into_view_if_needed'):
                    await element.scroll_into_view_if_needed()
                message = f"Scrolled to element: {step.target}"
            
            return ExecutionResult(
                step_number=step.step_number,
                action=step.action,
                success=True,
                message=message
            )
        except Exception as e:
            return ExecutionResult(
                step_number=step.step_number,
                action=step.action,
                success=False,
                message=f"Scroll failed: {str(e)}"
            )


class ValidatorAgent:
    """Agent responsible for validating test execution"""
    
    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
        self.system_prompt = """You are an expert QA validator. Your job is to analyze test execution 
        results and determine if the test passed or failed.
        
        Return your analysis in JSON format with NO trailing commas:
        {
            "overall_status": "PASS",
            "passed_steps": 5,
            "failed_steps": 0,
            "issues": [],
            "recommendations": [],
            "summary": "Brief summary"
        }"""
    
    def validate(self, test_plan: TestPlan, results: List[ExecutionResult]) -> Dict[str, Any]:
        """Validate test execution results"""
        
        results_summary = "\n".join([
            f"Step {r.step_number} ({r.action}): {'SUCCESS' if r.success else 'FAILED'} - {r.message}"
            for r in results
        ])
        
        prompt = f"""Analyze this test execution:

Test Objective: {test_plan.objective}

Success Criteria:
{chr(10).join(f"- {c}" for c in test_plan.success_criteria)}

Execution Results:
{results_summary}

Provide JSON with NO trailing commas:"""

        response = self.llm.generate(prompt, self.system_prompt, temperature=0.2)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                # Fix trailing commas
                json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                validation = json.loads(json_str)
            else:
                validation = json.loads(response)
            
            return validation
        except json.JSONDecodeError:
            passed = sum(1 for r in results if r.success)
            failed = len(results) - passed
            
            return {
                "overall_status": "PASS" if failed == 0 else "FAIL",
                "passed_steps": passed,
                "failed_steps": failed,
                "issues": [r.message for r in results if not r.success],
                "recommendations": ["Review failed steps"],
                "summary": f"{passed}/{len(results)} steps passed"
            }


class VibiumUITestingOrchestrator:
    """Main orchestrator for the multi-agent testing system with Vibium"""
    
    def __init__(self, ollama_model: str = "llama3.2:3b", headless: bool = False):
        self.llm_client = OllamaClient(model=ollama_model)
        self.planner = PlannerAgent(self.llm_client)
        self.executor = VibiumExecutorAgent(self.llm_client, headless=headless)
        self.validator = ValidatorAgent(self.llm_client)
    
    async def run_test(self, user_prompt: str) -> Dict[str, Any]:
        """Run complete test cycle: Plan -> Execute -> Validate"""
        
        print("=" * 80)
        print("UI TESTING AGENT SYSTEM (Vibium Edition)")
        print("=" * 80)
        
        # Step 1: Planning
        print("\n[PLANNER AGENT] Creating test plan...")
        test_plan = self.planner.create_plan(user_prompt)
        print(f"Objective: {test_plan.objective}")
        print(f"Steps: {len(test_plan.steps)}")
        for step in test_plan.steps:
            print(f"  {step.step_number}. {step.action} -> {step.target}")
        
        # Step 2: Execution
        print("\n[EXECUTOR AGENT] Executing test plan with Vibium...")
        results = []
        
        try:
            for step in test_plan.steps:
                print(f"\nExecuting Step {step.step_number}: {step.action} -> {step.target}")
                result = await self.executor.execute_step(step)
                results.append(result)
                
                status = "✓" if result.success else "✗"
                print(f"{status} {result.message}")
                
                # Stop on critical failures
                if not result.success and step.action in ["navigate"]:
                    print("Critical step failed, stopping execution")
                    break
        finally:
            await self.executor.close_browser()
        
        # Step 3: Validation
        print("\n[VALIDATOR AGENT] Validating results...")
        validation = self.validator.validate(test_plan, results)
        
        print(f"\nOverall Status: {validation['overall_status']}")
        print(f"Passed: {validation['passed_steps']}/{validation['passed_steps'] + validation['failed_steps']}")
        
        if validation.get('issues'):
            print("\nIssues Found:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        if validation.get('recommendations'):
            print("\nRecommendations:")
            for rec in validation['recommendations']:
                print(f"  - {rec}")
        
        print("\n" + "=" * 80)
        
        return {
            "test_plan": test_plan,
            "results": results,
            "validation": validation
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.executor.close_browser()


async def main():
    """Example usage with Vibium"""
    
    example_prompts = [
        "Test Google search: Navigate to google.com, search for 'Python programming', and verify results appear",
        "Test Google homepage: Open google.com and verify the search box is present",
    ]
    
    orchestrator = VibiumUITestingOrchestrator(
        ollama_model="llama3.2:3b",
        headless=False
    )
    
    try:
        user_prompt = example_prompts[0]
        print(f"\nRunning test: {user_prompt}\n")
        
        result = await orchestrator.run_test(user_prompt)
        
        print("\n\nTest Complete!")
        print(f"Final Status: {result['validation']['overall_status']}")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nError during test execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())