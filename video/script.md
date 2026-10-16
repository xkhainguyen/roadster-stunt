## hook
Tesla's Roadster demo was amazing: a kickflip over a semi truck, and a flip off a giant ramp.
I work on robot control, so I built my own version to show how a system like this could work.

## jump
First, most of the flight is a jump.
Fire the air jets down for half a second, and gravity does most of the rest.
Under three seconds in the air, over six meters up.

## planning
The hard part is the spin.
It has to start and stop at exactly the right moment.
So before the car moves, an optimizer uses a model of the car to plan the whole flight at once: every jet, every moment, using as little air as possible.
Each curve is one guess, and it keeps improving until the car clears the trucks and lands upright.

## hold
But a plan alone is fragile.
The real car never matches the model exactly: here it is a bit heavier, its jets are a bit weaker, and a gust hits it mid-flip.
Replay the same jet timing, and it hits the ground at eight meters per second.
So I add feedback: a controller called LQR measures the car's tilt a hundred times a second and fires the jets to correct it.

## landing
For the final moments, I switch to model predictive control.
Fifty times a second, it predicts the fall, shown in yellow, and picks the gentlest braking the jets can deliver.
Here I use TinyMPC, an open-source solver my collaborators and I built for small computers.

## ramp
The ramp stunt uses the same three tools.
Tesla says fans under the car suck it onto the ramp, so it can drive up past vertical.
Then the car leaves the ramp upside down, rolls upright while keeping its heading, and TinyMPC lands it.

## semi
And one stunt of my own: from the next lane, a kickflip onto a semi truck driving alongside, a bit like a rocket landing on a ship.
The same plan, feedback and landing put it down on a trailer only sixty centimeters wider than the car.

## end
Plan, correct, land.
This is my own implementation, not Tesla's. Huge respect to the Tesla and SpaceX teams who made it real.
