<script lang="ts" setup>
import { ref, onMounted, nextTick, computed, watch } from 'vue';
import { useI18n } from 'vue-i18n';

// this is the maximum value supported by the database for integer values
const MAX_INT = 2147483647;

const props = withDefaults(defineProps<{
  modelValue?: string | number;
  placeholder?: string;
  loading?: boolean;
  autofocus?: boolean;
  startIcon?: string;
  endIcon?: string;
  maxlength?: number;
  min?: number;
  max?: number;
  type?: 'text' | 'number' | 'password';
  rows?: number;
  autoResize?: boolean;
  maxHeight?: number;
  variant?: string;
  required?: boolean;
  disabled?: boolean;
  size?: 'normal' | 'large';
}>(), {min:0, max: MAX_INT, maxHeight: 150, size: 'normal'});

const emit = defineEmits<{
  (e: 'update:modelValue', value: string | number | undefined): void
  (e: 'keydown', event: KeyboardEvent): void
  (e: 'blur'): void
  (e: 'startIconClick'): void
  (e: 'endIconClick'): void
}>();

const { t } = useI18n();

const inputRef = ref<HTMLInputElement | HTMLTextAreaElement | null>(null);
const isOverMaxLength = ref(false);
const isUnderMin = ref(false);
const isOverMax = ref(false);
const isMissing = ref(false);
const tooltipRef = ref();

onMounted(() => {
  if (props.autofocus) {
    nextTick(() => {
      inputRef.value?.focus();
    });
  }

  autoResizeTextarea();
});

const autoResizeTextarea = () => {
  if (!inputRef.value || !props.autoResize || !props.rows) return;
  nextTick(() => {
    const textarea = inputRef.value as HTMLTextAreaElement;
    textarea.style.height = 'auto';
    const newHeight = Math.min(textarea.scrollHeight, props.maxHeight);
    textarea.style.height = `${newHeight}px`;
    textarea.style.overflowY = textarea.scrollHeight > props.maxHeight ? 'auto' : 'hidden';
  })
};

const updateValue = (event: Event) => {
  isMissing.value = false;
  isOverMaxLength.value = false;
  isUnderMin.value = false;
  isOverMax.value = false;
  let value = (event.target as HTMLInputElement | HTMLTextAreaElement).value;
  if (props.required && !value?.trim()) {
    isMissing.value = true;
  } else if (props.maxlength && value.length > props.maxlength) {
    isOverMaxLength.value = true;
  } else if (isNumber() && props.min && parseInt(value) < props.min) {
    isUnderMin.value = true;
  } else if (isNumber() && props.max && parseInt(value) > props.max) {
    isOverMax.value = true;
  }
  if (hasError()) {
    tooltipRef.value!.show(event);
  } else {
    tooltipRef.value!.hide();
  }
  emit('update:modelValue', (isNumber() && value === "") ? undefined: value);

  autoResizeTextarea();
};

const isNumber = () => props.type === 'number';

const hasError = () => isOverMaxLength.value || isUnderMin.value || isOverMax.value || isMissing.value

watch(() => props.required, (newRequired) => {
  isMissing.value = newRequired && (isNumber() ? !props.modelValue && props.modelValue !== 0 : !(props.modelValue as string)?.trim());
  if (!hasError()) {
    tooltipRef.value?.hide();
  }
});

const handleKeydown = (e: KeyboardEvent) => {
  emit('keydown', e);
};

const handleBlur = () => {
  emit('blur');
};

const onStartIconClick = () => {
  if (!props.disabled) emit('startIconClick');
};

const onEndIconClick = () => {
  if (!props.disabled) emit('endIconClick');
};

const inputIsAIGenerating = computed(() => {
  return props.loading && props.endIcon == "IconWand";
});

const sizeClasses = computed(() =>
  !props.rows && props.size === 'large' ? 'h-12 !text-base' : ''
);

const commonInputClasses = computed(() => [
  'w-full py-2 my-1 rounded-xl bg-surface placeholder:text-content-muted',
  'outline-1 outline-auxiliar-gray focus:outline-abstracta focus:ring-1 focus:ring-abstracta focus:border-abstracta',
  sizeClasses.value,
  props.startIcon ? 'px-11' : 'px-2',
  props.endIcon ? 'pr-11' : 'pr-2',
  inputIsAIGenerating.value && 'animate-glowing',
  isOverMaxLength.value && 'pr-10',
  hasError() && 'outline-error-alt focus:outline-error-alt ring-1 ring-error-alt',
  isNumber() && '[appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none',
  props.disabled && '!bg-surface-muted !text-content-muted opacity-60'
].filter(Boolean));

const commonIconClasses = computed(() => [
  'text-content-muted hover:text-content cursor-pointer absolute',
  props.disabled && '!text-content-muted opacity-60'
].filter(Boolean));

const translateClasses = computed(() => [
  'absolute',
  'top-1/2 -translate-y-1/2'
]);

const overMaxLength = computed(() => {
  if (!isNumber()) {
    const val = props.modelValue as string
    return props.maxlength && val && val.length > props.maxlength ? val.length - props.maxlength : 0
  }
});

watch(() => props.modelValue, () => {
  autoResizeTextarea();
});

defineExpose({
  focus: () => inputRef.value?.focus()
});
</script>

<template>
  <div class="relative">
    <!-- If rows available, we assume it's a textarea -->
    <template v-if="rows">
        <textarea
          ref="inputRef"
          :value="modelValue"
          @input="updateValue"
          @keydown="handleKeydown"
          @blur="handleBlur"
          :required="required"
          :placeholder="placeholder"
          :rows="rows"
          :disabled="disabled"
          :class="[
            commonInputClasses,
            autoResize ? 'resize-none' : 'resize-vertical'
          ]"
          :pattern="maxlength ? `.{0,${maxlength}}` : undefined">
        </textarea>
        <div v-if="endIcon && !isOverMaxLength"
          :class="[...commonIconClasses, 'bottom-5', 'right-2']"
          @click.stop="onEndIconClick">
          <component v-if="loading" is="IconLoader" class="animate-spin" />
          <component v-else :is="endIcon" />
        </div>
    </template>

    <template v-else>
      <slot name="start-icon" v-if="startIcon">
        <div :class="[...commonIconClasses, ...translateClasses, 'left-2']"
            @click.stop="onStartIconClick">
          <component v-if="loading" is="IconLoader" class="animate-spin" />
          <component v-else :is="startIcon" />
        </div>
      </slot>
      <input
        ref="inputRef"
        :value="modelValue"
        @input="updateValue"
        @keydown="handleKeydown"
        @blur="handleBlur"
        :required="required"
        :type="type || 'text'"
        :placeholder="placeholder"
        :disabled="disabled"
        :pattern="maxlength ? `.{0,${maxlength}}` : undefined"
        :class="[...commonInputClasses]"
      />
      <slot name="end-icon" v-if="endIcon && !isOverMaxLength">
        <div :class="[...commonIconClasses, ...translateClasses, 'right-2']"
            @click.stop="onEndIconClick">
          <component v-if="!startIcon && loading" is="IconLoader" class="animate-spin" />
          <component v-else :is="endIcon" />
        </div>
      </slot>
    </template>

    <span v-if="isOverMaxLength" :class="[...translateClasses, 'text-error-alt text-sm right-2']">-{{ overMaxLength }}</span>
    <Popover ref="tooltipRef">
      {{ t( isOverMaxLength ? 'overMaxLength' : isUnderMin ? 'underMin' : isOverMax ? 'overMax' : 'required', { overMaxAmount: overMaxLength, min, max }) }}
    </Popover>
  </div>
</template>

<i18n lang="json">
  {
    "en": {
      "overMaxLength": "You are {overMaxAmount} character/s over limit",
      "underMin": "The value must be greater than or equal to {min}",
      "overMax": "The value must be less than or equal to {max}",
      "required": "This field is required"
    },
    "es": {
      "overMaxLength": "Estás {overMaxAmount} caracter/es sobre el límite",
      "underMin": "El valor debe ser mayor o igual que {min}",
      "overMax": "El valor debe ser menor o igual que {max}",
      "required": "Este campo es requerido"
    }
  }
</i18n>
