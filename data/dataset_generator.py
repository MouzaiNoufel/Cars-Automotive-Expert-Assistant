"""
Automotive Dataset Generator
Creates high-quality instruction-response pairs for fine-tuning LLM on car knowledge.
"""

import random
from typing import List, Dict
import json


class AutomotiveDatasetGenerator:
    """Generates synthetic but realistic automotive instruction-response dataset."""
    
    def __init__(self, seed: int = 42):
        """Initialize dataset generator with random seed."""
        random.seed(seed)
        self.examples = []
    
    def generate_dataset(self, num_examples: int = 500) -> List[Dict[str, str]]:
        """
        Generate complete automotive dataset.
        
        Args:
            num_examples: Total number of examples to generate
            
        Returns:
            List of dicts with 'instruction', 'input', 'output' keys
        """
        self.examples = []
        
        # Calculate distribution across categories
        categories = [
            (self._generate_specification_examples, 0.15),  # 15%
            (self._generate_engine_examples, 0.12),  # 12%
            (self._generate_ev_hybrid_examples, 0.12),  # 12%
            (self._generate_comparison_examples, 0.12),  # 12%
            (self._generate_buying_advice_examples, 0.15),  # 15%
            (self._generate_maintenance_examples, 0.12),  # 12%
            (self._generate_safety_examples, 0.10),  # 10%
            (self._generate_technology_examples, 0.12),  # 12%
        ]
        
        for generator_func, proportion in categories:
            count = int(num_examples * proportion)
            self.examples.extend(generator_func(count))
        
        # Shuffle for variety
        random.shuffle(self.examples)
        
        return self.examples
    
    def _generate_specification_examples(self, n: int) -> List[Dict]:
        """Generate examples about car specifications."""
        templates = [
            {
                "instruction": "What does horsepower mean in a car?",
                "input": "",
                "output": "Horsepower (hp) measures an engine's power output - how much work it can do over time. One horsepower equals about 746 watts. Higher horsepower means faster acceleration and higher top speeds. For example, a 300hp sports car accelerates much quicker than a 150hp economy car. However, more power typically means lower fuel efficiency. For daily driving, 150-200hp is plenty; sports cars range from 300-700+hp."
            },
            {
                "instruction": "Explain what torque is and why it matters.",
                "input": "",
                "output": "Torque measures rotational force - the twisting power an engine produces. Measured in pound-feet (lb-ft) or Newton-meters (Nm). Torque determines how quickly a car accelerates from a stop and its towing capacity. High torque at low RPMs means strong low-end pull, great for trucks and everyday driving. Peak torque around 2000-3000 RPM is ideal for responsive acceleration. Diesel engines typically produce more torque than gasoline engines of similar size."
            },
            {
                "instruction": "What's the difference between horsepower and torque?",
                "input": "",
                "output": "Horsepower and torque measure different aspects of engine performance:\n\n**Torque**: The actual force the engine produces - how hard it pushes. Determines acceleration feel and towing ability.\n\n**Horsepower**: Torque × RPM ÷ 5252. Represents sustained power over time - determines top speed.\n\nThink of it this way: Torque gets you moving, horsepower keeps you going fast. A truck needs high torque for hauling; a sports car needs high horsepower for speed."
            },
            {
                "instruction": "What is curb weight and why does it matter?",
                "input": "",
                "output": "Curb weight is a vehicle's total weight with all fluids (fuel, oil, coolant) but no passengers or cargo. It matters because:\n\n1. **Performance**: Lighter cars accelerate faster and handle better\n2. **Fuel Economy**: Less weight = better MPG\n3. **Braking**: Heavier vehicles need longer stopping distances\n4. **Tire Wear**: More weight wears tires faster\n5. **Towing**: Affects towing capacity calculations\n\nModern cars use aluminum and high-strength steel to reduce weight while maintaining safety."
            },
            {
                "instruction": "Explain the 0-60 mph acceleration time.",
                "input": "",
                "output": "The 0-60 mph time measures how many seconds it takes a car to accelerate from a standstill to 60 mph. It's a key performance metric:\n\n**Economy Cars**: 8-10 seconds\n**Mid-size Sedans**: 6-8 seconds\n**Sports Cars**: 3-5 seconds\n**Supercars**: Under 3 seconds\n**Electric Vehicles**: Often 3-5 seconds (instant torque)\n\nFactors affecting 0-60 time: horsepower, torque, weight, drivetrain (AWD is faster), tire grip, and transmission type."
            },
        ]
        
        additional_specs = [
            ("What does engine displacement mean?", "", 
             "Engine displacement is the total volume of all cylinders, measured in liters (L) or cubic centimeters (cc). A 2.0L engine has 2000cc total cylinder volume. Larger displacement typically means more power and torque but lower fuel efficiency. Modern turbocharged engines produce similar power to larger engines with smaller displacement (e.g., a 2.0L turbo matching a 3.5L naturally aspirated engine). Displacement affects taxes and insurance in some countries."),
            
            ("What is compression ratio?", "",
             "Compression ratio measures how much the air-fuel mixture is compressed before ignition. Expressed as a ratio like 10:1. Higher compression (11:1 to 13:1) produces more power and efficiency but requires premium fuel to prevent knocking. Lower compression (8:1 to 10:1) runs on regular fuel but is less efficient. Turbocharged engines use lower compression ratios (9:1 to 10:1) since forced induction provides additional compression."),
            
            ("Explain wheel size numbers like 225/45R18.", "",
             "Tire size format: 225/45R18 breaks down as:\n\n**225**: Tire width in millimeters\n**45**: Aspect ratio (sidewall height as % of width) - 45% of 225mm\n**R**: Radial construction\n**18**: Wheel diameter in inches\n\nWider tires (higher first number) improve grip but reduce fuel economy. Lower aspect ratios (sportier, harsher ride) vs higher (more comfort). Always match the size recommended by the manufacturer for safety and performance."),
        ]
        
        examples = templates.copy()
        for instr, inp, out in additional_specs:
            examples.append({"instruction": instr, "input": inp, "output": out})
        
        # Replicate to reach desired count
        while len(examples) < n:
            examples.extend(templates)
        
        return examples[:n]
    
    def _generate_engine_examples(self, n: int) -> List[Dict]:
        """Generate examples about engine types and technology."""
        templates = [
            {
                "instruction": "What's the difference between a 4-cylinder and V6 engine?",
                "input": "",
                "output": "**4-Cylinder (Inline-4)**:\n- 4 cylinders in a straight line\n- Compact, lightweight, fuel-efficient\n- 150-250hp typically\n- Found in economy and mid-size cars\n- Smoother than 3-cylinder, more efficient than V6\n\n**V6**:\n- 6 cylinders in V-shaped configuration\n- More power (250-350hp typically)\n- Smoother operation, better for towing\n- Higher fuel consumption\n- Common in trucks, SUVs, performance sedans\n\nChoose 4-cylinder for efficiency, V6 for power and towing."
            },
            {
                "instruction": "Explain turbocharging and how it works.",
                "input": "",
                "output": "A turbocharger uses exhaust gases to spin a turbine that compresses incoming air, forcing more air (and fuel) into the engine. This increases power without increasing engine size.\n\n**Benefits**:\n- 30-40% more power from same displacement\n- Better fuel efficiency than larger naturally-aspirated engines\n- Reduced emissions\n\n**Drawbacks**:\n- Turbo lag (delay in power delivery)\n- More complex, higher maintenance costs\n- Requires premium fuel in many cases\n\nModern turbos with twin-scroll designs minimize lag significantly."
            },
            {
                "instruction": "What's the difference between turbocharged and supercharged engines?",
                "input": "",
                "output": "Both force more air into the engine for extra power, but differ in how they're powered:\n\n**Turbocharger**:\n- Driven by exhaust gases\n- More efficient (uses waste energy)\n- Turbo lag at low RPMs\n- Common in modern cars\n\n**Supercharger**:\n- Belt-driven by the engine\n- Instant power response (no lag)\n- Uses engine power to run (less efficient)\n- Popular in American muscle cars\n\nTurbos are more efficient, superchargers provide linear power delivery."
            },
            {
                "instruction": "How does a diesel engine differ from gasoline?",
                "input": "",
                "output": "**Diesel Engines**:\n- Compression ignition (no spark plugs)\n- Higher compression ratios (14:1 to 25:1)\n- More torque, better fuel economy (30-40% better)\n- Longer lifespan, lower RPM operation\n- Heavier, more expensive, louder\n\n**Gasoline Engines**:\n- Spark ignition\n- Lower compression (8:1 to 13:1)\n- Higher horsepower per liter\n- Lighter, quieter, smoother\n- Better cold-weather starting\n\nDiesels excel for towing and highway driving; gasoline for performance and city use."
            },
            {
                "instruction": "What is direct fuel injection?",
                "input": "",
                "output": "Direct injection sprays fuel directly into the combustion chamber at high pressure (2000+ psi), rather than into the intake manifold. \n\n**Advantages**:\n- Better fuel atomization and mixing\n- 10-15% improved fuel efficiency\n- More power from same displacement\n- Reduced emissions\n- Better throttle response\n\n**Disadvantages**:\n- Carbon buildup on intake valves\n- More expensive to maintain\n- Requires higher quality fuel\n\nMost modern engines use direct injection or a combination (port + direct injection) for optimal performance."
            },
        ]
        
        # Replicate to reach desired count
        examples = templates.copy()
        while len(examples) < n:
            examples.extend(templates)
        
        return examples[:n]
    
    def _generate_ev_hybrid_examples(self, n: int) -> List[Dict]:
        """Generate examples about electric and hybrid vehicles."""
        templates = [
            {
                "instruction": "How does a hybrid car work?",
                "input": "",
                "output": "A hybrid combines a gasoline engine with an electric motor and battery. The system automatically switches between or combines both power sources for optimal efficiency:\n\n**At Low Speeds**: Electric motor powers the car (zero emissions)\n**Accelerating**: Both engine and motor work together for extra power\n**Highway Cruising**: Gasoline engine takes over (most efficient)\n**Braking**: Regenerative braking captures energy to recharge battery\n\nThis results in 30-50% better fuel economy than gas-only vehicles, especially in city driving."
            },
            {
                "instruction": "What's the difference between hybrid, plug-in hybrid, and electric vehicles?",
                "input": "",
                "output": "**Hybrid (HEV)**:\n- Small battery charged only by engine and regenerative braking\n- Can't plug in to charge\n- 1-2 miles electric-only range\n- Example: Toyota Prius, Honda Accord Hybrid\n\n**Plug-In Hybrid (PHEV)**:\n- Larger battery that can be plugged in\n- 20-50 miles electric-only range\n- Gas engine for longer trips\n- Example: Toyota RAV4 Prime, Ford Escape PHEV\n\n**Electric Vehicle (BEV)**:\n- Battery-only, no gas engine\n- Must plug in to charge\n- 200-400+ mile range\n- Example: Tesla Model 3, Nissan Leaf\n\nPHEVs offer the most flexibility; BEVs the lowest operating costs."
            },
            {
                "instruction": "Explain how regenerative braking works.",
                "input": "",
                "output": "Regenerative braking converts the car's kinetic energy back into electrical energy when slowing down:\n\n1. When you brake, the electric motor reverses function and becomes a generator\n2. The motor's resistance slows the vehicle\n3. Generated electricity charges the battery\n4. Traditional friction brakes assist when needed\n\n**Benefits**:\n- Recovers 60-70% of braking energy\n- Extends driving range 10-15%\n- Reduces brake wear significantly\n- Provides smooth, controllable deceleration\n\nMost effective in stop-and-go city driving where there's frequent braking."
            },
            {
                "instruction": "What are the main advantages and disadvantages of electric vehicles?",
                "input": "",
                "output": "**Advantages**:\n- Zero tailpipe emissions\n- Lower operating costs (electricity cheaper than gas)\n- Less maintenance (no oil changes, fewer brake jobs)\n- Instant torque, smooth acceleration\n- Quieter operation\n- Can charge at home overnight\n\n**Disadvantages**:\n- Higher purchase price\n- Limited charging infrastructure in some areas\n- Longer \"refueling\" time (30 min to hours vs 5 min)\n- Range anxiety for long trips\n- Battery degradation over time\n- Limited model selection (improving rapidly)\n\nBest for: Regular commuters with home charging and a second car for road trips, or access to robust charging network."
            },
            {
                "instruction": "How long do electric car batteries last?",
                "input": "",
                "output": "Modern EV batteries are designed to last 10-20 years or 100,000-200,000 miles:\n\n**Degradation Rate**: Expect 1-2% capacity loss per year, so 70-80% capacity remains after 10 years.\n\n**Factors Affecting Lifespan**:\n- Extreme temperatures (hot climates degrade faster)\n- Frequent DC fast charging\n- Regularly charging to 100% or depleting to 0%\n- High mileage\n\n**Best Practices**:\n- Keep charge between 20-80% for daily use\n- Avoid frequent fast charging when possible\n- Park in shade/garage in hot weather\n- Use battery thermal management system\n\nMost manufacturers warranty batteries for 8 years/100,000 miles."
            },
        ]
        
        examples = templates.copy()
        while len(examples) < n:
            examples.extend(templates)
        
        return examples[:n]
    
    def _generate_comparison_examples(self, n: int) -> List[Dict]:
        """Generate vehicle comparison examples."""
        templates = [
            {
                "instruction": "Compare the Toyota Camry and Honda Accord.",
                "input": "I need a reliable family sedan with good fuel economy.",
                "output": "Both are excellent choices with similar strengths:\n\n**Toyota Camry**:\n- Slightly better reliability record\n- More conservative styling\n- Softer, more comfortable ride\n- Stronger hybrid option (52 MPG)\n- Better resale value\n\n**Honda Accord**:\n- More engaging driving dynamics\n- Sportier styling\n- More upscale interior feel\n- More powerful base engine (192hp vs 203hp)\n- Better infotainment system\n\n**Verdict**: Both get 30+ MPG, have strong reliability, and low maintenance costs. Choose Camry for maximum reliability and comfort; Accord for driving enjoyment and features. You can't go wrong with either."
            },
            {
                "instruction": "Compare sedans vs SUVs for a family.",
                "input": "We have two kids and a dog.",
                "output": "**Sedans (e.g., Honda Accord, Toyota Camry)**:\n- Better fuel economy (28-32 MPG)\n- Lower purchase price ($28k-35k)\n- Easier to park and maneuver\n- Smoother ride on highways\n- Lower center of gravity (better handling)\n\n**SUVs (e.g., Honda CR-V, Toyota RAV4)**:\n- More cargo space (especially with seats down)\n- Higher seating position (better visibility)\n- Better for road trips with gear\n- AWD available for snow\n- Easier to load kids/car seats\n\n**For Two Kids + Dog**: A mid-size SUV like CR-V or RAV4 is ideal. More space for sports equipment, groceries, and dog crate. Slightly worse MPG (27-30) but much more practical for family life."
            },
            {
                "instruction": "Compare FWD, RWD, and AWD.",
                "input": "",
                "output": "**Front-Wheel Drive (FWD)**:\n- Power to front wheels\n- Better fuel economy, lower cost\n- More interior space (no driveshaft tunnel)\n- Good traction in snow (weight over drive wheels)\n- Examples: Honda Civic, Toyota Camry\n\n**Rear-Wheel Drive (RWD)**:\n- Power to rear wheels\n- Better weight distribution, handling dynamics\n- Preferred for sports cars and trucks\n- Can be tricky in snow\n- Examples: BMW 3 Series, Ford F-150\n\n**All-Wheel Drive (AWD)**:\n- Power to all four wheels (automatic)\n- Best traction in rain/snow\n- Slightly lower MPG, higher cost\n- Examples: Subaru Outback, Audi A4\n\nChoose FWD for efficiency, RWD for performance, AWD for all-weather capability."
            },
            {
                "instruction": "Gas vs diesel for a truck?",
                "input": "I need a truck for towing a camper and daily driving.",
                "output": "**Gasoline Truck** (e.g., Ford F-150 5.0L V8):\n- Lower purchase price ($5k-10k less)\n- Quieter, smoother operation\n- Cheaper maintenance\n- Better for short trips and city driving\n- Adequate for light towing (7,000-11,000 lbs)\n\n**Diesel Truck** (e.g., Ford F-250 6.7L Power Stroke):\n- Much better towing capacity (15,000-20,000+ lbs)\n- 30-40% better fuel economy when towing\n- Longer engine lifespan (300k+ miles)\n- Higher torque at low RPMs (perfect for towing)\n- Higher initial cost and maintenance\n\n**For Towing Camper**: If your camper is over 8,000 lbs or you tow frequently, diesel is worth it. Better fuel economy while towing and more capability. For occasional towing of smaller campers, a modern gas V8 is sufficient and cheaper."
            },
            {
                "instruction": "Compare luxury brands: BMW, Mercedes, Audi.",
                "input": "",
                "output": "**BMW**:\n- Focus: Driving dynamics and performance\n- Best handling in class, \"Ultimate Driving Machine\"\n- Sportier steering, suspension tuning\n- Rear-wheel drive standard (better balance)\n- Slightly less reliable, higher maintenance costs\n\n**Mercedes-Benz**:\n- Focus: Luxury and comfort\n- Superior interior materials and build quality\n- Smoothest ride, best for highway cruising\n- Latest technology and safety features\n- Strong brand prestige\n\n**Audi**:\n- Focus: Technology and all-weather capability\n- Quattro AWD system (excellent traction)\n- Most advanced infotainment (MMI system)\n- Understated, clean design\n- Middle ground between BMW sport and Mercedes luxury\n\nChoose BMW for driving enthusiasts, Mercedes for luxury, Audi for technology and AWD."
            },
        ]
        
        examples = templates.copy()
        while len(examples) < n:
            examples.extend(templates)
        
        return examples[:n]
    
    def _generate_buying_advice_examples(self, n: int) -> List[Dict]:
        """Generate car buying advice examples."""
        templates = [
            {
                "instruction": "Best used car under $15,000?",
                "input": "I need something reliable with low running costs.",
                "output": "For $15,000, focus on these reliable options:\n\n**1. Honda Civic (2015-2017)**:\n- Excellent reliability and resale value\n- 30+ MPG, low insurance costs\n- Modern safety features\n- Available: $12k-15k with 60k-80k miles\n\n**2. Toyota Corolla (2014-2016)**:\n- Best-in-class dependability\n- Cheap parts and maintenance\n- Boring but bulletproof\n- Available: $11k-14k with 70k-90k miles\n\n**3. Mazda3 (2016-2018)**:\n- Fun to drive, upscale interior\n- Good fuel economy (28-32 MPG)\n- More features for the price\n- Available: $13k-15k with 50k-70k miles\n\n**4. Honda CR-V (2012-2014)**:\n- If you need SUV space\n- Honda reliability, practical\n- Available: $14k-15k with 80k-100k miles\n\n**Tips**: Get pre-purchase inspection, check CARFAX, verify maintenance records, budget $500-1000 for immediate repairs."
            },
            {
                "instruction": "Should I buy new or used?",
                "input": "I want to keep the car for 10+ years.",
                "output": "**Buy New If**:\n- You plan to keep it 10+ years (you get full lifespan)\n- You want latest safety tech and fuel efficiency\n- You drive a lot (warranty coverage for early miles)\n- You can get 0-2% financing\n- You want to avoid unknown maintenance history\n\n**Buy Used (2-3 Years Old) If**:\n- You want to avoid 20-30% depreciation hit\n- You're okay with previous generation tech\n- You're handy with maintenance\n- You can pay cash or get reasonable financing\n\n**For 10+ Year Ownership**: New makes sense. Total cost is similar since:\n- No early repairs from wear\n- Full warranty coverage (3-5 years)\n- You control all maintenance from day one\n- No hidden issues from previous owner\n- Latest fuel efficiency saves money long-term\n\nRecommended: Buy new Toyota/Honda/Mazda and maintain religiously."
            },
            {
                "instruction": "What should I look for when buying a used car?",
                "input": "",
                "output": "**Before Seeing the Car**:\n1. Check CARFAX/AutoCheck for accidents, title issues\n2. Verify ownership history (one-owner is best)\n3. Confirm maintenance records available\n4. Research common problems for that model/year\n\n**During Inspection**:\n1. Look for mismatched paint (accident repairs)\n2. Check tire wear (uneven = alignment/suspension issues)\n3. Test all electronics, A/C, windows, lights\n4. Look under car for leaks, rust\n5. Check oil color (dark/dirty = poor maintenance)\n\n**Test Drive**:\n1. Listen for unusual noises (transmission, suspension)\n2. Test brakes in empty lot (should be firm, straight)\n3. Check alignment (car pulls left/right?)\n4. Test acceleration (smooth or hesitation?)\n\n**Essential**: Get pre-purchase inspection by independent mechanic ($100-150). They'll catch issues you'd miss. Negotiate repairs or walk away if major problems found."
            },
            {
                "instruction": "Is it worth buying a certified pre-owned car?",
                "input": "",
                "output": "**Certified Pre-Owned (CPO) Benefits**:\n- Extended warranty (typically 1 year/12k miles added)\n- 100+ point inspection by dealer\n- Roadside assistance\n- Return period (often 7 days)\n- Better financing rates\n- No major accident history guaranteed\n\n**Drawbacks**:\n- 10-15% higher price than non-CPO equivalent\n- Still a used car with existing wear\n- Warranty less comprehensive than new car\n\n**Worth It If**:\n- The extra $1500-2500 gives you peace of mind\n- You're buying European luxury (expensive repairs)\n- The CPO warranty is transferable (helps resale)\n- You can't afford new but want warranty protection\n\n**Skip CPO If**:\n- You're buying a reliable brand (Honda, Toyota)\n- You found a well-maintained private party sale\n- You're mechanically inclined\n- The CPO premium is excessive (>$3000)\n\n**Best Value**: CPO luxury cars (BMW, Mercedes, Audi) where repair costs are high."
            },
            {
                "instruction": "What's the best car for a new driver?",
                "input": "My teenager just got their license.",
                "output": "**Key Criteria**:\n1. Top safety ratings (IIHS Top Safety Pick)\n2. Good visibility, easy to maneuver\n3. Reliable, cheap to maintain\n4. Modest power (avoid sports cars)\n5. Affordable insurance\n\n**Top Choices**:\n\n**1. Honda Civic (2015-2018)**:\n- Excellent safety scores\n- Easy to drive, good visibility\n- 30+ MPG, low running costs\n- Strong reliability\n- Used: $12k-16k\n\n**2. Mazda3 (2014-2017)**:\n- Great handling teaches good habits\n- Advanced safety features\n- Fun but not overpowered\n- Used: $10k-14k\n\n**3. Subaru Impreza (2015-2017)**:\n- AWD for all-weather safety\n- Excellent crash test ratings\n- Good visibility\n- Used: $13k-17k\n\n**Avoid**: Sports cars, large trucks/SUVs (harder to maneuver), anything with 250+ hp, vehicles without ESC/ABS.\n\n**Pro Tip**: Buy used but invest in ADAS features (automatic emergency braking, blind spot monitoring) - they prevent accidents."
            },
        ]
        
        examples = templates.copy()
        while len(examples) < n:
            examples.extend(templates)
        
        return examples[:n]
    
    def _generate_maintenance_examples(self, n: int) -> List[Dict]:
        """Generate maintenance and repair advice."""
        templates = [
            {
                "instruction": "How often should I change my car's oil?",
                "input": "I drive a 2020 Honda Civic with synthetic oil.",
                "output": "For a 2020 Honda Civic with synthetic oil:\n\n**Recommended Interval**: Every 7,500-10,000 miles or 12 months (whichever comes first)\n\nHonda's Maintenance Minder system will alert you when it's due based on:\n- Driving conditions\n- Engine temperature cycles\n- Mileage\n\n**Change Earlier If**:\n- Frequent short trips (under 10 miles)\n- Extreme temperatures\n- Dusty/dirty environments\n- Towing or heavy loads\n- Stop-and-go traffic daily\n\n**Why Synthetic Lasts Longer**:\n- Better heat resistance\n- Fewer impurities\n- Maintains viscosity longer\n- Superior engine protection\n\n**Cost**: $60-80 at dealer, $40-50 at quick-lube shop. Never exceed 10,000 miles even with synthetic."
            },
            {
                "instruction": "What are the most important maintenance tasks?",
                "input": "",
                "output": "**Critical Maintenance (Don't Skip)**:\n\n1. **Oil Changes** (every 5k-10k miles):\n   - Prevents engine wear and failure\n   \n2. **Tire Rotations** (every 5k-7k miles):\n   - Even wear, extends tire life 25%\n   \n3. **Brake Inspections** (every 10k miles):\n   - Catch issues before complete failure\n   \n4. **Air Filter** (every 15k-30k miles):\n   - Improves MPG and engine performance\n   \n5. **Coolant Flush** (every 30k-50k miles):\n   - Prevents overheating and engine damage\n   \n6. **Transmission Fluid** (every 30k-60k miles):\n   - Extends transmission life (expensive to replace)\n   \n7. **Timing Belt** (every 60k-100k miles if applicable):\n   - Failure causes catastrophic engine damage\n\n**Follow your owner's manual schedule religiously. Skipping maintenance costs far more in repairs later.**"
            },
            {
                "instruction": "How long do brake pads typically last?",
                "input": "",
                "output": "**Average Lifespan**: 25,000-70,000 miles depending on driving style and conditions.\n\n**Factors Affecting Life**:\n\n**Shorter Life (25k-40k)**:\n- City driving with frequent stops\n- Aggressive braking\n- Mountain/hilly terrain\n- Heavy vehicle (truck, SUV)\n- Towing frequently\n\n**Longer Life (50k-70k)**:\n- Highway driving\n- Gentle braking habits\n- Flat terrain\n- Lighter vehicle\n- Using engine braking\n\n**Warning Signs**:\n- Squealing/squeaking noises\n- Grinding sound (metal-on-metal - get fixed NOW)\n- Pulsating brake pedal\n- Longer stopping distances\n- Brake warning light\n\n**Cost**: $150-300 per axle for pads and labor. Rotors add $200-400 if needed.\n\n**Pro Tip**: Inspect brakes every oil change. Replace pads before they damage rotors (saves money)."
            },
            {
                "instruction": "When should I replace my tires?",
                "input": "",
                "output": "**Replace When**:\n\n1. **Tread Depth Below 4/32\"**:\n   - Legal minimum is 2/32\", but unsafe in rain\n   - Use penny test: Insert penny with Lincoln's head down\n   - If you see top of his head, replace tires\n   \n2. **Age Over 6 Years**:\n   - Rubber degrades even with good tread\n   - Check DOT code on sidewall for manufacture date\n   \n3. **Visible Damage**:\n   - Cracks, bulges, or cuts in sidewall\n   - Uneven wear patterns\n   - Exposed cords or belts\n\n**Average Lifespan**:\n- Standard tires: 40,000-60,000 miles\n- Performance tires: 25,000-40,000 miles\n- All-season: 50,000-70,000 miles\n- Truck tires: 40,000-80,000 miles\n\n**Extend Tire Life**:\n- Rotate every 5,000-7,000 miles\n- Maintain proper inflation (check monthly)\n- Get alignment checked if car pulls\n- Avoid aggressive acceleration/braking\n\n**Cost**: $400-800 for 4 tires (mid-range brands)"
            },
            {
                "instruction": "What's the difference between regular and synthetic oil?",
                "input": "",
                "output": "**Conventional Oil**:\n- Refined from crude petroleum\n- Contains more impurities\n- Breaks down faster (3k-5k mile changes)\n- Cheaper ($25-40 per change)\n- Fine for older, simple engines\n\n**Synthetic Oil**:\n- Engineered molecules, uniform consistency\n- Fewer impurities, better additives\n- Lasts 7,500-10,000 miles\n- More expensive ($50-80 per change)\n- Better protection, especially in extremes\n\n**Advantages of Synthetic**:\n- Better flow in cold weather (easier starts)\n- Resists breakdown at high temperatures\n- Improves fuel economy (1-2%)\n- Reduces engine wear\n- Cleaner engine (less sludge)\n\n**Who Needs Synthetic**:\n- Turbocharged engines (required)\n- High-performance vehicles\n- Extreme climates\n- Towing/hauling regularly\n- Want extended oil change intervals\n\n**Verdict**: Synthetic costs 50% more but lasts 100% longer. Total cost is similar, but better for your engine. Worth it for most modern cars."
            },
        ]
        
        examples = templates.copy()
        while len(examples) < n:
            examples.extend(templates)
        
        return examples[:n]
    
    def _generate_safety_examples(self, n: int) -> List[Dict]:
        """Generate safety feature and advice examples."""
        templates = [
            {
                "instruction": "What are the most important safety features in a new car?",
                "input": "",
                "output": "**Essential Safety Features (Must-Haves)**:\n\n1. **Electronic Stability Control (ESC)**: Prevents skidding and loss of control\n2. **Anti-lock Brakes (ABS)**: Maintains steering during emergency braking\n3. **Airbags**: Front, side, curtain (8+ airbags ideal)\n4. **Backup Camera**: Now mandatory, prevents backing accidents\n\n**Highly Recommended (ADAS)**:\n\n5. **Automatic Emergency Braking (AEB)**: Stops car if collision imminent\n6. **Blind Spot Monitoring**: Alerts to vehicles in blind spots\n7. **Lane Departure Warning/Assist**: Prevents drifting out of lane\n8. **Adaptive Cruise Control**: Maintains safe following distance\n\n**Nice to Have**:\n\n9. **360-degree Camera**: Better visibility when parking\n10. **Rear Cross-Traffic Alert**: Warns when backing out\n\nStudies show AEB alone reduces rear-end crashes by 50%. Invest in these technologies - they save lives."
            },
            {
                "instruction": "How do I check if a car has been in an accident?",
                "input": "",
                "output": "**Online Reports**:\n1. **CARFAX** ($40): Most comprehensive, shows accidents, service records, ownership\n2. **AutoCheck** ($25): Similar to CARFAX, sometimes catches different info\n3. **NMVTIS** ($5-10): Government database, basic but cheap\n\n**Physical Inspection**:\n1. **Paint Inconsistencies**: Look for color mismatches, orange peel texture, overspray\n2. **Panel Gaps**: Uneven spacing between body panels\n3. **Doors/Hood Alignment**: Should close smoothly and evenly\n4. **Welding Marks**: Check under hood and trunk for non-factory welds\n5. **Undercarriage**: Look for frame damage, bent components\n6. **Airbag Light**: If illuminated, may have been in accident\n\n**Test Drive**:\n- Car pulls to one side (frame damage)\n- Unusual vibrations\n- Uneven tire wear\n\n**Pro Tip**: Use a magnet on body panels. If it doesn't stick, there's Bondo (filler) underneath - accident repair.\n\n**Always** get pre-purchase inspection by mechanic. They have tools to detect frame damage CARFAX might miss."
            },
            {
                "instruction": "What does the IIHS Top Safety Pick award mean?",
                "input": "",
                "output": "The Insurance Institute for Highway Safety (IIHS) **Top Safety Pick** is the gold standard for vehicle safety.\n\n**Criteria for Award**:\n1. **Crashworthiness**: \"Good\" ratings in all 6 crash tests:\n   - Moderate overlap front\n   - Small overlap front (driver and passenger)\n   - Side impact\n   - Roof strength\n   - Head restraints\n\n2. **Crash Avoidance**: \"Advanced\" or \"Superior\" rating for front crash prevention\n\n3. **Headlights**: \"Acceptable\" or \"Good\" rating\n\n**Top Safety Pick+** (highest award):\n- Same as above but requires \"Good\" or \"Acceptable\" headlights on all trims\n\n**Why It Matters**:\n- Independent, rigorous testing\n- More stringent than federal safety standards\n- Insurance companies offer discounts\n- Best indicator of real-world crash protection\n\n**2024 Winners Include**: Honda Accord, Toyota Camry, Mazda CX-5, Subaru Outback, Genesis GV70\n\nWhen car shopping, prioritize IIHS Top Safety Pick+ winners for maximum protection."
            },
            {
                "instruction": "Explain how adaptive cruise control works.",
                "input": "",
                "output": "**Adaptive Cruise Control (ACC)** automatically maintains a safe following distance from the car ahead.\n\n**How It Works**:\n1. **Radar/Camera Sensors**: Detect vehicles ahead and measure distance/speed\n2. **Set Speed**: You choose cruise speed (e.g., 70 mph)\n3. **Automatic Adjustment**: \n   - If road is clear, maintains your set speed\n   - If slower car ahead, slows to match their speed\n   - Maintains preset following distance (2-4 seconds)\n   - When clear, accelerates back to your speed\n\n**Advanced Features**:\n- **Stop-and-Go**: Works in traffic, brings car to complete stop\n- **Lane Centering**: Keeps car centered in lane\n- **Speed Limit Recognition**: Adjusts to posted speed limits\n\n**Benefits**:\n- Reduces driver fatigue on long trips\n- Smoother acceleration/braking (better MPG)\n- Prevents tailgating\n- Foundation for semi-autonomous driving\n\n**Limitations**:\n- Requires clear sensor view (doesn't work in heavy rain/snow)\n- May not detect stationary objects\n- Driver must stay alert (not self-driving)\n\nHighway driving game-changer. Reduces stress and improves safety significantly."
            },
            {
                "instruction": "What is a 5-star safety rating?",
                "input": "",
                "output": "The **5-Star Safety Rating** comes from the National Highway Traffic Safety Administration (NHTSA) and indicates top crash protection.\n\n**Rating System** (1-5 stars):\n- **5 Stars**: Best (10% or less injury risk)\n- **4 Stars**: Good (11-20% injury risk)\n- **3 Stars**: Average (21-35% injury risk)\n- **2 Stars**: Below average (36-45% injury risk)\n- **1 Star**: Poor (46%+ injury risk)\n\n**Test Categories**:\n1. **Frontal Crash**: Vehicle hits barrier at 35 mph\n2. **Side Crash**: Pole and moving barrier tests\n3. **Rollover**: Resistance to rolling over\n\n**Overall Rating**: Combination of all three tests\n\n**Additional Ratings**:\n- Front crash prevention (automatic braking)\n- Pedestrian detection\n\n**5-Star vs IIHS Top Safety Pick**:\n- NHTSA (government): Crash test focus\n- IIHS (insurance industry): More comprehensive, includes crash avoidance\n\nBoth are important. Best cars earn 5-star NHTSA + IIHS Top Safety Pick+.\n\nExamples: Honda Accord, Toyota Camry, Tesla Model 3, Mazda CX-5"
            },
        ]
        
        examples = templates.copy()
        while len(examples) < n:
            examples.extend(templates)
        
        return examples[:n]
    
    def _generate_technology_examples(self, n: int) -> List[Dict]:
        """Generate automotive technology examples."""
        templates = [
            {
                "instruction": "What's the difference between AWD and 4WD?",
                "input": "",
                "output": "**All-Wheel Drive (AWD)**:\n- Power automatically distributed to all 4 wheels\n- Computer-controlled, always active\n- Best for on-road traction (rain, snow)\n- Found in crossovers, sedans, wagons\n- Examples: Subaru Outback, Audi Quattro\n\n**Four-Wheel Drive (4WD)**:\n- Driver manually engages 4WD (usually)\n- Has low-range gearing for off-road\n- Better for serious off-roading, rock crawling\n- Found in trucks and rugged SUVs\n- Examples: Jeep Wrangler, Ford F-150\n\n**Key Differences**:\n- AWD: Automatic, on-road focused, no low range\n- 4WD: Manual control, off-road capability, low range for crawling\n\n**Fuel Economy**: Both reduce MPG by 1-2 vs 2WD\n\n**Choose AWD** for daily driving in snow/rain. **Choose 4WD** for off-road adventures, deep snow, towing in rough terrain."
            },
            {
                "instruction": "How does a CVT transmission work?",
                "input": "",
                "output": "A **Continuously Variable Transmission (CVT)** uses belts and pulleys instead of fixed gears for infinite ratio variations.\n\n**How It Works**:\n1. Two cone-shaped pulleys connected by metal belt\n2. Pulleys change diameter to vary gear ratio\n3. Computer adjusts ratios seamlessly based on speed/load\n4. No gear shifts - smooth, continuous acceleration\n\n**Advantages**:\n- Better fuel economy (3-5% vs traditional automatic)\n- Smooth acceleration, no shift shock\n- Engine stays at optimal RPM\n- Lighter and more compact\n\n**Disadvantages**:\n- \"Rubber band\" feel (engine revs high, acceleration feels disconnected)\n- Less engaging for driving enthusiasts\n- Some reliability concerns (older models)\n- Expensive to repair\n\n**Found In**: Honda CR-V, Toyota Corolla, Nissan Altima, Subaru models\n\n**Modern Improvements**: Simulated gear shifts, better programming make newer CVTs feel more natural.\n\n**Verdict**: Great for fuel efficiency and daily commuting; not for performance driving."
            },
            {
                "instruction": "What is start-stop technology?",
                "input": "",
                "output": "**Start-Stop** (also called Auto Start-Stop) automatically shuts off the engine when the car is stopped, then restarts it when you lift your foot off the brake.\n\n**How It Works**:\n1. Come to complete stop at red light\n2. Engine shuts off after 1-2 seconds\n3. All accessories (A/C, radio) continue running on battery\n4. Lift foot from brake or press gas\n5. Engine instantly restarts (0.3-0.4 seconds)\n\n**Benefits**:\n- Improves fuel economy 3-10% (best in city driving)\n- Reduces emissions during idling\n- Quieter at stoplights\n- Extends engine life (less idle time)\n\n**Concerns**:\n- Starter and battery wear (though designed for it)\n- Slight delay when restarting\n- Can be annoying in stop-and-go traffic\n\n**Most systems allow you to disable it** (button near shifter)\n\n**Found In**: Most new cars from 2018+\n\n**Reality**: Saves 0.5-1 gallon per tank in city driving. Small benefit but adds up over time. The system is designed for hundreds of thousands of cycles - reliability is not an issue on modern cars."
            },
            {
                "instruction": "Explain dual-clutch transmission (DCT).",
                "input": "",
                "output": "A **Dual-Clutch Transmission (DCT)** combines the efficiency of manual transmission with the convenience of automatic.\n\n**How It Works**:\n1. Two separate clutches:\n   - Clutch 1: Odd gears (1, 3, 5, 7)\n   - Clutch 2: Even gears (2, 4, 6)\n2. Next gear is pre-selected on the idle clutch\n3. During shift, clutches swap almost instantly (0.1 seconds)\n4. Computer controls everything\n\n**Advantages**:\n- Lightning-fast shifts (faster than manual)\n- No power interruption during shifts\n- Better fuel economy than traditional automatic\n- More engaging driving experience\n- Can handle high performance/torque\n\n**Disadvantages**:\n- Jerky at low speeds (especially in traffic)\n- Complex, expensive to repair\n- Some models have reliability issues\n- Not great for towing (can overheat)\n\n**Found In**: Volkswagen/Audi (DSG), Porsche (PDK), Hyundai/Kia (DCT)\n\n**Best For**: Performance driving and highway use. Avoid for heavy city traffic unless it's a refined modern system (2018+)."
            },
            {
                "instruction": "What is torque vectoring?",
                "input": "",
                "output": "**Torque Vectoring** intelligently distributes power between wheels to improve handling, especially during cornering.\n\n**How It Works**:\n1. Sensors monitor steering angle, speed, lateral G-forces\n2. System detects understeer or oversteer\n3. Redirects power (and applies braking) to specific wheels\n4. Outside rear wheel gets more power during turns\n5. Car rotates better, corners flatter and faster\n\n**Types**:\n\n**Brake-Based** (common):\n- Uses brakes to slow inside wheel\n- More affordable\n- Examples: Most AWD systems\n\n**Active Differential** (advanced):\n- Mechanically varies power to each wheel\n- More precise, better performance\n- Examples: BMW M cars, Audi Sport models\n\n**Benefits**:\n- Sharper turn-in\n- Reduced understeer\n- Better traction out of corners\n- Increased stability\n- More confidence-inspiring driving\n\n**Found In**: Performance AWD vehicles, some FWD hot hatches\n\n**Real-World**: Makes the car feel more agile and planted. Especially beneficial in AWD performance cars (Audi RS, BMW M, AMG) and on wet/slippery roads."
            },
        ]
        
        examples = templates.copy()
        while len(examples) < n:
            examples.extend(templates)
        
        return examples[:n]
    
    def save_to_json(self, filepath: str):
        """Save generated dataset to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.examples, f, indent=2, ensure_ascii=False)
        print(f"✓ Dataset saved to {filepath} ({len(self.examples)} examples)")


if __name__ == "__main__":
    # Example usage
    generator = AutomotiveDatasetGenerator(seed=42)
    dataset = generator.generate_dataset(num_examples=500)
    generator.save_to_json("automotive_dataset.json")
    
    # Print sample
    print("\n" + "="*70)
    print("SAMPLE EXAMPLES:")
    print("="*70)
    for i, example in enumerate(dataset[:3], 1):
        print(f"\nExample {i}:")
        print(f"Instruction: {example['instruction']}")
        if example['input']:
            print(f"Input: {example['input']}")
        print(f"Output: {example['output'][:200]}...")
