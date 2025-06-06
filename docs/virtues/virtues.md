## Usability/Simplicity
Usability of software is directly tied to the effectiveness of the user’s ability to use the software. Manuals and walkthroughs help address gaps in usability but greatly compromise maintainability. Intuitive interfaces address the root issue. Keeping the software as simple as possible helps remove unnecessary complexity. Learnability can be described as the difficulty for a user to learn the software and can be coerced as a secondary virtue. It can also refer to the software itself learning.

The natural enemies of usability are originality, urgency, and versatility. New innovative features are rarely understood enough to be simplified and showcase a natural tension between these virtues. Urgency leads to rushing or even skipping user testing. Versatility invites ambiguity that may undermine intuitiveness.

>Usability is a top priority for the BT Servant, especially given the intended user base. The app must be easy to use, which is why leveraging WhatsApp is essential. While additional functionality beyond simple Q&A may be valuable, the experience should always feel as effortless as chatting with a friend. Interacting with the BT Servant should be no more difficult than having a normal conversation on WhatsApp.

## Originality/Innovation
Innovation in software can sometimes steal the limelight and temptingly overshadow other virtues. Original ideas are how most projects start and make it easier to sell the project to investors. These new approaches are usually what gives the projects their intrinsic value.

The natural enemies of originality are usability and stability. Many times usability and stability suffer at first but as the approach is tested and usage is studied, both can be improved.

>As of 6/5/2025, we are still in the early stages of development. At this point, originality and innovation are especially valuable. However, we’ve intentionally focused our efforts on a specific area: the challenges surrounding timely context retrieval (RAG, broadly defined) and the subsequent generation of relevant and accurate responses. While many others are also working in this space, innovation will likely come through borrowing, combining, and recombining existing ideas rather than through major breakthroughs — though that remains to be seen.

## Stability/Reliability
Stability covers the typical freezes and crashes associated with software bugs. Reliability is associated with the ability to recover without the loss of data. These may be treated as one or as two primary virtues depending on the need. There may be times where reliability is mandatory but crashes are acceptable.

Stability’s natural enemies are urgency and originality. Rushing something out of the door increases the chances for bugs. New innovative approaches are inherently less tested in the real world and consequences are unknown.

>As we approach the end of the POC phase—or the very beginning of the prototype phase, depending on how you look at it—these concerns are currently low priority. At this stage, if something fails on the Twilio or Fly.io side, the user simply doesn’t receive a response in WhatsApp (though the error is logged). As we move further into the prototype phase, we’ll need to:
(1) ensure a message is always returned to the user, even if it’s just an error notification; and
(2) deliver responses in a timely manner (though what qualifies as "timely" still needs to be defined).

## Urgency/Timeliness
Urgency ensures that the project is delivered in time to meet the users need before they solve their issues another way. In production software, the tyranny of the urgent needs to be kept in check. In preproduction software, it may not be prioritized enough. Some users cannot wait for releases and move ahead on a similar path that might not be parallel. This can lead to more work of interoperability if they choose other software in their workflow. Rushing to meet their timeline may sacrifice other priorities. Harness the missed opportunity to study the use case to help a future user base.

Urgency’s natural enemies are maintainability, efficiency, and usability. Rushing leaves little to no time for ensuring that code is either maintainable or efficient. User testing also takes time and may lead to redesign.

## Efficiency/Scalability

Efficiency can keep hardware costs down as more work can be done on less hardware. As long as the software performs at a pace that keeps up with users development time is wasted on optimization. Efficiency is a never-ending quest. There are always ways to increase performance and architect a better solution. This is what many developers value most. Scalability is a form of efficiency under changing load and can prevent wasted hardware resources from going idle while not under load. Serverless computing is a great example of this. It is easy to think this is only a concern of developers and engineers but needs to be prioritized by all stakeholders.

Efficiency’s natural enemies are urgency, maintainability, and versatility. Many times refactors sacrifice one or two of these.

> We've intentionally kept this a low priority given how early we are in the project. Nevertheless, response time—specifically the 18-second delay between message send and BT Servant reply (observed at its worst in Colombia, although this demo was called a smashing success)—was raised during a recent demo. Even at this early stage, we’re being pushed to consider what users expect in terms of response time when interacting with a domain-expert bot. We’ll need to define that expectation more rigorously through user testing and other methods to determine where to draw the line in this area.

## Maintainability/Manageability
Maintainability is widely underappreciated and covers a wide gamut of secondary virtues. This can include code understandability, modularity, testability, and hire-ability, among others. Primarily this focuses on the project transcending a few contributors holding the keys to the project. This also affects the ability to add more team members to the same project. Understandability allows other developers to quickly read and intuitively know what the code does and how to work with it. Modularity allows more parts of the project to be worked on simultaneously as well as reuse of code. Testability can help verify that the existing code acts as intended and ensures future changes are less likely to break previously working code. Hire-ability is assessing how likely your choices affect finding a qualified developer to work on this project. Avoid obscure languages and frameworks unless the use case demands it.

The natural enemies of maintainability are urgency and efficiency. Rushing to get features out the door or obsessing over performance typically trap a project from achieving long-term maintainability. Technical debt is a common side effect of not prioritizing this virtue.

## Adaptability/Versatility
Adaptability is the ability to adapt or be adapted to multiple use cases. There is a balance to applying this versatility. Attempting to make software that is too versatile can reduce its effectiveness for any single use case. Designing software workflows to handle too few use cases limit the usability in related use cases. Studying related use cases help identify areas of natural versatility without sacrificing focus or effectiveness.

The natural enemies of versatility are usability and efficiency. The more a usability workflow and code efficiency are optimized for a particular use case the less versatile it becomes.

## Interoperability
Interoperability is the ability to import, export, and use data between separate software systems. This can increase applicable use cases and market share for users that already rely on or would like to integrate other related software in their workflow.

The natural enemies of interoperability are originality and urgency. Innovation can be stifled if too much focus is placed on interoperability. Rushing to meet deadlines can sometimes lead to skipping the study and integration of existing standards.

> As we move forward, we’ll need to consider what kind of ecosystem the BT Servant should participate in—or put differently, how it might serve other systems with related use cases. While interoperability is not a current priority, we are intentionally avoiding design decisions that would preclude this virtue in the future.

## Affordability/Sustainability
Affordability is addressed from two angles. The price of the software to the user and the development team. Sustainability is the intersection of the two. If a team can not afford to deliver the software from the income of the software, they must fund the project by another means. This is common in open source software and can be done in different ways. Assessing the costs ensures pet projects are kept in check and that each project’s invested effort is proportionate to the vision and mission of the team.

Affordability’s natural enemies are nearly every other virtue as they each incur a cost

## Reality Assessment
All of the above virtues have to be balanced with reality. Some problems just take time to solve especially to accomplish it in a stable way. Since software development takes time it also costs money. Inexperienced users may take time to learn a new piece of software and initial usability testing might be misleading. Choosing the wrong userbase during testing may lead to ineffective software for the intended users. Is it realistic to or advantageous to focus on interoperability? Does this project really need maintainability or is it a small enough scope to rewrite it if there is no one who can maintain it? Reality helps ensure costs are assessed and that there is enough budgeted to complete the project.