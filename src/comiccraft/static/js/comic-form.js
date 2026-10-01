const comicForm = document.querySelector("#comic-form");

if (comicForm) {
  comicForm.querySelectorAll(".combo-field").forEach((combo) => {
    const input = combo.querySelector("[role='combobox']");
    const toggle = combo.querySelector(".combo-toggle");
    const list = combo.querySelector("[role='listbox']");
    const options = [...list.querySelectorAll("[role='option']")];
    let activeOption = -1;

    const setActiveOption = (index) => {
      activeOption = index;
      options.forEach((option, optionIndex) => {
        option.setAttribute("aria-selected", String(optionIndex === activeOption));
      });

      if (activeOption >= 0) {
        input.setAttribute("aria-activedescendant", options[activeOption].id);
        options[activeOption].scrollIntoView({ block: "nearest" });
      } else {
        input.removeAttribute("aria-activedescendant");
      }
    };

    const setExpanded = (expanded) => {
      list.hidden = !expanded;
      input.setAttribute("aria-expanded", String(expanded));
      toggle.setAttribute("aria-expanded", String(expanded));
      if (!expanded) {
        setActiveOption(-1);
      }
    };

    const showAllOptions = () => {
      options.forEach((option) => {
        option.hidden = false;
      });
      setExpanded(true);
      const currentValue = options.findIndex(
        (option) => option.dataset.value === input.value,
      );
      setActiveOption(currentValue);
    };

    const filterOptions = (query) => {
      const normalizedQuery = query.trim().toLocaleLowerCase();
      options.forEach((option) => {
        option.hidden = !option.textContent.trim().toLocaleLowerCase().includes(normalizedQuery);
      });
      setExpanded(true);
      setActiveOption(-1);
    };

    const chooseOption = (option) => {
      input.value = option.dataset.value;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
      setExpanded(false);
      input.focus();
    };

    const visibleOptions = () => options.filter((option) => !option.hidden);

    const moveActiveOption = (direction) => {
      const visible = visibleOptions();
      if (!visible.length) {
        return;
      }

      const currentIndex = visible.indexOf(options[activeOption]);
      const nextIndex = currentIndex < 0
        ? (direction > 0 ? 0 : visible.length - 1)
        : (currentIndex + direction + visible.length) % visible.length;
      setActiveOption(options.indexOf(visible[nextIndex]));
    };

    input.addEventListener("focus", showAllOptions);
    input.addEventListener("click", showAllOptions);
    input.addEventListener("input", () => filterOptions(input.value));
    input.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        if (list.hidden || !visibleOptions().length) {
          showAllOptions();
        }
        moveActiveOption(event.key === "ArrowDown" ? 1 : -1);
      } else if (event.key === "Enter" && !list.hidden && activeOption >= 0) {
        event.preventDefault();
        chooseOption(options[activeOption]);
      } else if (event.key === "Escape" && !list.hidden) {
        event.preventDefault();
        setExpanded(false);
      } else if (event.key === "Tab") {
        setExpanded(false);
      }
    });

    toggle.addEventListener("pointerdown", (event) => event.preventDefault());
    toggle.addEventListener("click", () => {
      if (list.hidden || options.some((option) => option.hidden)) {
        input.focus();
        showAllOptions();
      } else {
        setExpanded(false);
      }
    });

    options.forEach((option) => {
      option.addEventListener("pointerdown", (event) => event.preventDefault());
      option.addEventListener("click", () => chooseOption(option));
    });

    document.addEventListener("pointerdown", (event) => {
      if (!combo.contains(event.target)) {
        setExpanded(false);
      }
    });
  });

  let submissionInProgress = false;

  comicForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (submissionInProgress) {
      return;
    }

    submissionInProgress = true;
    const submitButton = comicForm.querySelector("button[type='submit']");
    const buttonLabel = submitButton.querySelector(".generate-button-label");
    const buttonDots = submitButton.querySelector(".generate-button-dots");
    const stages = {
      preparing: "Preparing story",
      story: "Generating 5-panel story",
      narration: "Generating narration and dialogue",
      images: "Creating illustrations",
      panels: "Creating comic panels",
      assembly: "Assembling comic",
      preview: "Preparing preview",
      pdf: "Preparing PDF",
    };
    let dotCount = 0;
    let resultHtml = null;
    let generationError = null;
    document.querySelector(".error-banner")?.remove();
    const dotAnimation = window.setInterval(() => {
      dotCount = (dotCount % 3) + 1;
      buttonDots.textContent = ".".repeat(dotCount);
    }, 350);

    const setStage = (label) => {
      buttonLabel.textContent = label;
    };

    submitButton.disabled = true;
    submitButton.setAttribute("aria-busy", "true");
    submitButton.classList.add("is-generating");
    buttonDots.hidden = false;
    setStage(stages.preparing);

    try {
      const response = await fetch(comicForm.action, {
        method: "POST",
        body: new FormData(comicForm),
        headers: { Accept: "text/event-stream" },
      });
      if (!response.ok || !response.body) {
        throw new Error("The progress stream could not be opened.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let streamEnded = false;
      while (!streamEnded) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const frames = buffer.split(/\r?\n\r?\n/);
        buffer = frames.pop();
        for (const frame of frames) {
          const event = frame.match(/^event: (.+)$/m)?.[1];
          const dataLine = frame.match(/^data: (.*)$/m)?.[1];
          if (!event || dataLine === undefined) {
            continue;
          }
          const data = JSON.parse(dataLine);
          if (stages[event]) {
            setStage(stages[event]);
          } else if (event === "panel") {
            setStage(`Creating Panel ${data.panel_number} of 5`);
          } else if (event === "complete") {
            window.clearInterval(dotAnimation);
            buttonDots.hidden = true;
            submitButton.classList.remove("is-generating");
            setStage("✓ Comic ready");
          } else if (event === "result") {
            resultHtml = data.html;
            break;
          } else if (event === "error") {
            generationError = data;
          }
        }
        if (done) {
          streamEnded = true;
        }
      }

      if (generationError) {
        const errorBanner = document.querySelector(".error-banner") || document.createElement("div");
        errorBanner.className = "error-banner";
        errorBanner.setAttribute("role", "alert");
        errorBanner.textContent = generationError.message;
        if (!errorBanner.isConnected) {
          comicForm.before(errorBanner);
        }
        window.clearInterval(dotAnimation);
        submitButton.disabled = false;
        submitButton.removeAttribute("aria-busy");
        submitButton.classList.remove("is-generating");
        buttonLabel.textContent = "Generate comic";
        buttonDots.hidden = true;
        submissionInProgress = false;
        return;
      }

      if (!resultHtml) {
        throw new Error("The generation response ended unexpectedly.");
      }
      window.clearInterval(dotAnimation);
      window.requestAnimationFrame(() => {
        document.open();
        document.write(resultHtml);
        document.close();
      });
    } catch (error) {
      window.clearInterval(dotAnimation);
      submitButton.disabled = false;
      submitButton.removeAttribute("aria-busy");
      submitButton.classList.remove("is-generating");
      buttonLabel.textContent = "Generate comic";
      buttonDots.hidden = true;
      const errorBanner = document.querySelector(".error-banner") || document.createElement("div");
      errorBanner.className = "error-banner";
      errorBanner.setAttribute("role", "alert");
      errorBanner.textContent = "Something went wrong while creating your comic. Please try again.";
      if (!errorBanner.isConnected) {
        comicForm.before(errorBanner);
      }
      submissionInProgress = false;
    }
  });
}
